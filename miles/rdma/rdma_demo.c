// rdma_demo.c — minimal RDMA SEND/RECV demo using librdmacm.
//   server:  ./rdma_demo
//   client:  ./rdma_demo <server-ip>
//
// librdmacm's rdma_cm API handles the painful part of RDMA — the QP state
// machine and address/route resolution — so this stays short while still
// exercising the real verbs data path (PD, MR, QP, CQ, work requests).
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#define PORT "7471"
#define MSG  "hello from RDMA"

// Fail loudly: an RDMA call that silently no-ops is the worst kind of bug.
#define CHECK(cond, msg) do { if (!(cond)) { perror(msg); exit(1); } } while (0)

// One QP handshake supports exactly one outstanding send and one recv here.
static struct ibv_qp_init_attr qp_attr(void) {
    struct ibv_qp_init_attr a = {0};
    a.cap.max_send_wr  = a.cap.max_recv_wr  = 1;
    a.cap.max_send_sge = a.cap.max_recv_sge = 1;
    a.cap.max_inline_data = 128;   // small payloads ride inline, no MR needed
    a.sq_sig_all = 1;              // signal every send so we get a completion
    return a;
}

static void run_server(void) {
    struct rdma_addrinfo hints = {0}, *res;
    struct rdma_cm_id *listen_id, *id;
    struct ibv_qp_init_attr attr = qp_attr();
    struct ibv_wc wc;
    char buf[128] = {0};

    hints.ai_flags = RAI_PASSIVE;  // this side binds and listens
    CHECK(rdma_getaddrinfo(NULL, PORT, &hints, &res) == 0, "getaddrinfo");
    CHECK(rdma_create_ep(&listen_id, res, NULL, &attr) == 0, "create_ep");
    CHECK(rdma_listen(listen_id, 1) == 0, "listen");
    printf("server: listening on port %s\n", PORT);

    // Block until a client arrives; `id` is the per-connection endpoint.
    CHECK(rdma_get_request(listen_id, &id) == 0, "get_request");

    // Register the receive buffer, then post the recv BEFORE accepting so the
    // NIC has somewhere to land the incoming message the instant it connects.
    struct ibv_mr *mr = rdma_reg_msgs(id, buf, sizeof buf);
    CHECK(mr != NULL, "reg_msgs");
    CHECK(rdma_post_recv(id, NULL, buf, sizeof buf, mr) == 0, "post_recv");
    CHECK(rdma_accept(id, NULL) == 0, "accept");

    // Poll the completion queue until the receive lands.
    while (rdma_get_recv_comp(id, &wc) == 0)
        ;
    printf("server: received \"%s\" (%u bytes)\n", buf, wc.byte_len);

    rdma_disconnect(id);
    rdma_dereg_mr(mr);
    rdma_destroy_ep(id);
    rdma_destroy_ep(listen_id);
    rdma_freeaddrinfo(res);
}

static void run_client(const char *host) {
    struct rdma_addrinfo hints = {0}, *res;
    struct rdma_cm_id *id;
    struct ibv_qp_init_attr attr = qp_attr();
    struct ibv_wc wc;
    char buf[128];

    CHECK(rdma_getaddrinfo(host, PORT, &hints, &res) == 0, "getaddrinfo");
    CHECK(rdma_create_ep(&id, res, NULL, &attr) == 0, "create_ep");
    CHECK(rdma_connect(id, NULL) == 0, "connect");
    printf("client: connected to %s\n", host);

    strncpy(buf, MSG, sizeof buf);
    struct ibv_mr *mr = rdma_reg_msgs(id, buf, sizeof buf);
    CHECK(mr != NULL, "reg_msgs");
    // IBV_SEND_INLINE copies the payload into the WR itself — lowest latency
    // for tiny messages, and it means the MR isn't strictly required.
    CHECK(rdma_post_send(id, NULL, buf, strlen(buf) + 1, mr, IBV_SEND_INLINE) == 0,
          "post_send");
    while (rdma_get_send_comp(id, &wc) == 0)
        ;
    printf("client: sent \"%s\"\n", buf);

    rdma_disconnect(id);
    rdma_dereg_mr(mr);
    rdma_destroy_ep(id);
    rdma_freeaddrinfo(res);
}

int main(int argc, char **argv) {
    if (argc > 1)
        run_client(argv[1]);
    else
        run_server();
    return 0;
}
