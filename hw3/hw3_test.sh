#!/bin/sh

mkdir -p model/cnn53
cd model/cnn53
curl -O 'https://dl.dropboxusercontent.com/s/kq7ng6iggjtypbx/model'
cd ../..
python3 cnn53.py test $1 $2
