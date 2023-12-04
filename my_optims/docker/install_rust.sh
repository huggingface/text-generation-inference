set -e
echo "install rust"
tar -zxvf rust-1.72.0-x86_64-unknown-linux-gnu.tar.gz
cd rust-1.72.0-x86_64-unknown-linux-gnu
./install.sh
cd -
