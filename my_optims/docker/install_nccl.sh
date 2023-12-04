set -e
echo "install nccl"
cp libnccl.so.2.17.1 /usr/lib/x86_64-linux-gnu/
cp nccl_net.h /usr/include/
cp nccl.h /usr/include/
ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2.17.1 /usr/lib/x86_64-linux-gnu/libnccl.so
ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2.17.1 /usr/lib/x86_64-linux-gnu/libnccl.so.2
