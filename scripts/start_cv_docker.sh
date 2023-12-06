docker run -itd --name gamebet --entrypoint bash --network host \
    --shm-size 10g --memory 10g --volume /data/k12ai:/data/k12ai \
    --workdir /data/k12ai/codes/hzcsbet
    --runtime nvidia hzcsai_com/k12cv
