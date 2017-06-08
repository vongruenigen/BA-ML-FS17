#/bin/bash


for dir in `ls`
do
  cd $dir
  unzip *.zip
  cd ..
done