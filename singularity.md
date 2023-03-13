# Build the image
- Go to a preprocess node:
```bash
srun --partition=prepost -A vfy@v100 --pty bash
```
- Load singularity module
```bash
module load singularity
```

- Go to $WORK partition and make a directory there.
```bash
cd $WORK
mkdir tmp
```

- Build the image from online repository
```bash
SINGULARITY_TMPDIR=`pwd`/tmp singularity build h3.sif docker://flbbb/jz-ssm:latest
```
