set -e
echo "install openmpi"
tar -xjvf openmpi-4.1.6.tar.bz2
cd openmpi-4.1.6/
./configure --prefix=/usr/local
make all
make install
cd - 
