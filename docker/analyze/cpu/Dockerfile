FROM plato3d/plato-spack:cpu

RUN spack install exo2obj ^seacas@master~mpi~x11
RUN spack install platoanalyze~cuda~amgx @develop build_type=Debug ^trilinos build_type=Debug ^omega-h build_type=Debug ^platoengine build_type=Debug
RUN echo 'spack load platoanalyze' >> /etc/profile.d/spack.sh

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
