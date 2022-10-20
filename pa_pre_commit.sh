#!/bin/sh

required_version=10

if command -v clang-format
then 
  version=$(clang-format --version | awk -F 'version' '{print $2}' | awk '{print $1}' | awk -F '.' '{print $1}')

  if [ $version -ge $required_version ]
  then
    echo Formatting files in commit using clang-format
    for FILE in $(git diff --cached --name-only | grep -E '.*\.(c|cpp|h|hpp)\b')
    do
      clang-format -i $FILE
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
