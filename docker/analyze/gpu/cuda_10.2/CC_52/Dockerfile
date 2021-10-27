FROM plato3d/plato-spack:cuda-10.2

RUN spack install exo2obj ^seacas@master~mpi~x11
RUN spack install platoanalyze+cuda+mpmd+python+esp @develop ^trilinos cuda_arch=52 ^amgx cuda_arch=52 ^cuda@10.2.89
RUN echo 'spack load platoanalyze' >> /etc/profile.d/spack.sh
RUN echo 'spack load exo2obj' >> /etc/profile.d/spack.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
