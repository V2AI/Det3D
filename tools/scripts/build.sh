sudo apt install libboost-dev

pip install -r requirements.txt

# pushd lib/deps/spconv
# python setup.py bdist_wheel
# pushd dist
# pip install *.whl
# popd
# rm -rf build dist *.egg*
# popd
#
# pushd lib/deps/apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
# rm -rf build dist *.egg*
# popd

pushd lib/core/csrc/pointnet2 
python setup.py install
rm -rf build dist *.egg*
popd

pushd lib/core/csrc/rroi_align
python setup.py install
rm -rf build dist *.egg*
popd

pushd lib/core/csrc/iou3d
python setup.py install
rm -rf build dist *.egg*
popd

pushd lib/core/csrc/roipool3d
python setup.py install
rm -rf build dist *.egg*
popd


pushd lib/core/csrc/alignfeature
python setup.py install
rm -rf build dist *.egg*
popd


pushd lib/core/csrc/correlation
python setup.py install
rm -rf build dist *.egg*
popd
