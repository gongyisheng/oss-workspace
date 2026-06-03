# Map GPU UUID -> index (e.g. GPU-abcd... -> 0).
declare -A gpu_index
while IFS=',' read -r idx uuid; do
  gpu_index[$(echo "$uuid" | tr -d ' ')]=$(echo "$idx" | tr -d ' ')
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

# Map host PID -> GPU index and memory for every GPU compute process.
declare -A pid_gpu pid_mem
while IFS=',' read -r pid mem uuid; do
  pid=$(echo "$pid" | tr -d ' ')
  mem=$(echo "$mem" | grep -oE '[0-9]+')
  uuid=$(echo "$uuid" | tr -d ' ')
  pid_gpu[$pid]=${gpu_index[$uuid]:-?}
  pid_mem[$pid]=$mem
done < <(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader)

# For each running container, report the GPU usage of its processes.
for c in $(docker ps -q); do
  name=$(docker inspect --format '{{.Name}}' "$c" | sed 's#^/##')
  cpids=$(docker top "$c" -eo pid 2>/dev/null | tail -n +2)
  for gp in "${!pid_gpu[@]}"; do
    if grep -qx "$gp" <<<"$cpids"; then
      printf '%-25s pid=%-8s gpu=%-3s mem=%sMiB\n' \
        "$name" "$gp" "${pid_gpu[$gp]}" "${pid_mem[$gp]}"
    fi
  done
done | sort