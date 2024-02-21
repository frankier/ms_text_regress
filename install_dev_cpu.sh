poetry export --without-hashes -E demos -E datasets -E eval -E viewer --with=dev -o tmp-requirements.txt

sed -i '/^torch\b/d' tmp-requirements.txt
echo 'torch --extra-index-url https://download.pytorch.org/whl/cpu' >> tmp-requirements.txt

sed -i '/^accelerate\b/d' tmp-requirements.txt
echo '-e git+https://github.com/frankier/accelerate.git@73ccf16c3cf1c9c3d0c196d11d9c5b14a0520758#egg=accelerate' >> tmp-requirements.txt

poetry run pip install -r tmp-requirements.txt
rm tmp-requirements.txt
