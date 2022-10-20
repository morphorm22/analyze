#!/bin/sh

if command -v clang-format
then 
  echo Formatting files in commit using clang-format
  for FILE in $(git diff --cached --name-only | grep -E '.*\.(c|cpp|h|hpp)\b')
  do
      clang-format -i $FILE
      git add $FILE
  done
  echo Formatted files have been commited
else
  echo Error: clang-format was not found. Aborting commit
  exit 1
fi


