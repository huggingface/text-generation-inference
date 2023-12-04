set -e
echo "install tgi"
python -m pip install Ninja 
cd text-generation-inference/my_optims/nccl_test
python setup.py install 
cd -
cd text-generation-inference/
make install-launcher
cp -r server/ /opt/conda/lib/python3.9/site-packages/text_generation_server/
cd -
