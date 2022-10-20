#!/bin/sh

required_version=10

found=0
if (command -v clang-format) 
then 
  exe=clang-format
  found=1
elif (command -v clang-format-$required_version)
then
  exe=clang-format-$required_version
  found=1
fi

if (found==1)
then 
  version=$($exe --version | awk -F 'version' '{print $2}' | awk '{print $1}' | awk -F '.' '{print $1}')

  if [ $version -ge $required_version ]
  then
    echo Formatting files in commit using clang-format
    for FILE in $(git diff --cached --name-only | grep -E '.*\.(c|cpp|h|hpp)\b')
    do
      $exe -i $FILE
      git add $FILE
    done
    echo Formatted files have been commited
  else
    echo Error: clang-format version does not meet requirement. Must use version $required_version or greater. Aborting commit
    exit 1
  fi
else
  echo Error: clang-format was not found. Aborting commit
  exit 1
fi
