#!/bin/bash

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J generate
#! Which project should be charged:
#SBATCH -A JONSSON-SL3-CPU
#SBATCH -p icelake
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*76)
#! The Ice Lake (icelake) nodes have 76 CPUs (cores) each and
#! 3380 MiB of memory per CPU.
#SBATCH --ntasks=3
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Run job as test
#SBATCH --qos=intr

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by cpu number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*76 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 CPU by default, and each CPU is allocated 3380 MiB
#! of memory. If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MiB per node).

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

module load vtk/7.1.1
source /rds/user/ke330/hpc-work/hpc-env/bin/activate

#! We don't need this currently as we just run hardcoded command
#! (But it was part of the default setup)

# #! Run options for the application:
# python_script="/rds/user/ke330/hpc-work/generate_simulated_data.py"

# #! Define the command-line arguments
# exec_dir="/rds/user/ke330/hpc-work/tubulaton/bin/"
# exec_file_name="programme"
# input_mesh_dir="/rds/user/ke330/hpc-work/tubulaton/structures/2024_Karan/"
# output_dir="/rds/user/ke330/hpc-work/SimulatedData/"
# sample_name="$SLURM_ARRAY_TASK_ID"

# #! Full path to application executable: 
# application="python3"
# options="$python_script $exec_dir $exec_file_name $input_mesh_dir $output_dir $sample_name"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 76:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
#CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):

#! We have used hardcoded commands instead
#CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

#! Instead of eval a CMD we just run a (hardcoded) command directly
#eval $CMD 

# #! Run options for the application:
# python_script="/rds/user/ke330/hpc-work/generate_simulated_data.py"

# #! Define the command-line arguments
exec_dir="/rds/user/ke330/hpc-work/tubulaton/bin/"
input_mesh_dir="/rds/user/ke330/hpc-work/tubulaton/structures/2024_Karan/"
output_dir="/rds/user/ke330/hpc-work/SimulatedData/"
sample_name="$SLURM_ARRAY_TASK_ID"

python3 /rds/user/ke330/hpc-work/generation/config_setup.py $exec_dir $input_mesh_dir $output_dir $TIME_STEPS $sample_name

#TODO - when we use the new tubulaton version 
# programme will need to be replaced with tubulaton
/rds/user/ke330/hpc-work/tubulaton/bin/programme /rds/user/ke330/hpc-work/tubulaton/bin/config.ini

#To prevent conflicts between VTK and python vtk library
module unload vtk/7.1.1

tubulaton_dir="${output_dir}tubulaton_dir"
tubulaton_file_name="tubulaton-${sample_name}_${TIME_STEPS}.vtk"
tubulaton_file_path="$tubulaton_dir/$tubulaton_file_name"

python3 /rds/user/ke330/hpc-work/generation/generate.py $output_dir $sample_name $tubulaton_file_path

# #! Full path to application executable: 
# application="python3"
# options="$python_script $exec_dir $exec_file_name $input_mesh_dir $output_dir $sample_name"

