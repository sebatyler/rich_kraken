#!/bin/bash

files=".env.prod credentials/firebase_service_account.json"

echo "다음 파일들을 업로드하시겠습니까?"
echo "--------------------------------"
for file in $files; do
    echo "- $file"
done
echo "--------------------------------"

while true; do
    read -p "계속 진행하시겠습니까? (yes/no) " confirm
    case $confirm in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "yes 또는 no를 입력해주세요.";;
    esac
done

for file in $files; do
    aws --profile sebatyler s3 cp $file s3://sebatyler-dev/rich_trader/$file
done
