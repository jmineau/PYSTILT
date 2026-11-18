In Containers (Kubernetes)
==========================

A project can run in many containers at once, such as the pods of a
Kubernetes Job, without Slurm. Each container runs one share of the
releases:

.. code-block:: bash

   stilt run /project --task 3/20

``--task I/N`` runs task ``I`` of ``N`` in this process: every ``N``-th
release of the project, starting at ``I`` (0 to ``N - 1``). Every task
splits the releases the same way, whenever it starts. Releases whose
results exist are skipped, so running a task again only fills in what is
missing.

``--releases ids.txt`` runs only the releases listed in a file, one id per
line. With ``--task`` too, the task takes its share of that list.

What every container needs
--------------------------

- PYSTILT installed (``pip install pystilt``), and HYSPLIT's ``hycs_std``
  on ``PATH`` or in the HYSPLIT folder ``directory`` names
  (:doc:`../getting_started/installation`). Every container needs the same
  ``hycs_std``: its checksum is part of each run's settings, so another
  build writes to another settings folder. A container that only reports
  status needs none.
- The project folder (``config.yaml`` and ``releases.csv``), the
  meteorology, and the output directory, on a volume all the containers
  share. Results are written there, one file per simulation, so the
  containers do not need to talk to each other. The output directory can
  instead be on an object store (``output: s3://bucket/...``, see
  :doc:`projects`), with the containers given its credentials.
  ``stilt run --output s3://bucket/...`` uses another output directory
  than ``config.yaml`` names, without editing it.
- A compute root, where each run gets its workdir. Set
  ``PYSTILT_COMPUTE_ROOT`` to a folder on fast local disk, such as an
  ``emptyDir``.

.. _fetch-once:

Fetching the meteorology once
-----------------------------

With ``download`` set, a run downloads the met files its hours need into the
met ``directory``. Pods on one shared volume find the files the first pod
fetched. Pods with their own disks do not: 50 pods whose releases fall on
the same days download, and crop, the same files 50 times.

A task (``stilt run --task``) never fetches for the others, so
``stilt fetch`` gets the files once, before the Job starts (``stilt submit``
does this for you on Slurm, :ref:`slurm-fetch`):

.. code-block:: bash

   stilt fetch /project

For each met it gets the files that cover the time of every simulation that
has yet to run, cropped to the met's subgrid. They are the files ``stilt
run`` would ask for, so a run that follows finds them all and downloads
nothing. Run it on one pod, with the met ``directory`` on the volume the
Job's pods then mount. Simulations that are finished need no met and are
left out; ``--output`` is as for ``stilt run``, so it reads the same
results.

A met without ``download`` is only checked: ``stilt fetch`` reports the
files it lacks, as ``stilt check`` does, and does not make them.

It exits with:

- 0 when every file is there
- 1 when the archive does not have some files; it prints the times of the
  runs that need them
- 2 when the command line is wrong or a met directory cannot be read
- 3 when it was stopped (Ctrl-C, or the SIGTERM that Kubernetes and Slurm
  send), or the archive was busy or could not be reached, which a retry may
  get past
- 4 for any other unexpected error (:ref:`exit-codes`)

Run it again to get the missing files or to finish after a stop: the files
it got are kept. A stop removes the file being downloaded. A process killed
with SIGKILL, which cannot be caught, leaves a hidden
``.<name>.<random>.partial`` file in the met directory; delete it.

Fetches can overlap. Several ``stilt fetch`` processes, or a fetch beside
runs, can share one met directory: each file is written under a temporary
name and renamed into place when it is whole, so no one reads half a file.
Two that start the same file both download it, and the second to finish
replaces the first with the same contents.

In Python, ``project.fetch()`` does the same and returns the files, one row
each, with the times it could not get files for.

A Kubernetes indexed Job
------------------------

An indexed Job gives each pod its index in ``JOB_COMPLETION_INDEX``, which
is the task number:

.. code-block:: yaml

   apiVersion: batch/v1
   kind: Job
   metadata:
     name: stilt-my-project
   spec:
     completionMode: Indexed
     completions: 20          # N, the number of tasks
     parallelism: 20          # pods running at once
     backoffLimit: 6
     template:
       spec:
         restartPolicy: Never
         containers:
           - name: stilt
             image: my-registry/pystilt:latest   # any image with PYSTILT
             command: ["sh", "-c"]
             args: ["exec stilt run /project --task $JOB_COMPLETION_INDEX/20 --cpus 4"]
             env:
               - name: PYSTILT_COMPUTE_ROOT
                 value: /scratch
             resources:
               requests:
                 cpu: "4"
                 memory: 8Gi
             volumeMounts:
               - {name: project, mountPath: /project}
               - {name: met, mountPath: /met, readOnly: true}
               - {name: scratch, mountPath: /scratch}
         volumes:
           - name: project
             persistentVolumeClaim: {claimName: stilt-project}
           - name: met
             persistentVolumeClaim: {claimName: hrrr-arl}
           - name: scratch
             emptyDir: {}

The met ``directory`` in ``config.yaml`` must be the path inside the
container (``/met`` here). ``completions`` and the ``N`` in ``--task`` must
match. ``--cpus 4`` runs four releases at once in each pod. ``exec`` makes
``stilt`` the process that receives Kubernetes' stop signal, so a stopped
pod stops its runs cleanly and exits with code 3, within a few seconds of
the signal and so inside Kubernetes' default 30 s grace period. That holds
for ``--cpus`` above 1 too: the pod's workers and HYSPLIT runs stop with it.

.. _exit-codes:

Exit codes
----------

``stilt run`` exits with:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Code
     - Meaning
   * - 0
     - Every simulation it ran is complete.
   * - 1
     - Some simulations failed. ``stilt status`` counts them by reason. A
       retry runs them again, which helps only when the cause is fixed.
   * - 2
     - The command line was wrong: an unknown option, a bad value, or a
       directory that is not a project. Or the project's files are invalid:
       ``config.yaml`` is not valid YAML or has a misspelled setting, or a
       row of ``releases.csv`` is bad. Or a met directory does not exist
       or holds no ARL file (``stilt check`` lists it; ``stilt status`` exits
       with 2 for it only when the output may hold results). Nothing ran. Do
       not retry.
   * - 3
     - Retry it. Some simulations did not finish, because the run was
       stopped (Ctrl-C, or a stop signal from Kubernetes or Slurm), a worker
       process was killed (``--cpus`` above 1; the error names the releases
       that were in progress, and it is usually the out-of-memory killer, so a
       retry may need fewer ``--cpus``), or a connection or I/O error, such as
       an output bucket that could not be reached, stopped an unfinished
       simulation or ended the run. A retry continues where it stopped.
   * - 4
     - A bug: any other error the command did not expect. Its traceback is
       printed. Do not retry; a retry meets the same error.

An unexpected error ends with 3, not 4, when a retry may get past it:

- a refused, reset or timed-out connection, or a name that did not resolve,
  including the cloud libraries' own errors for these (botocore's
  ``EndpointConnectionError`` and ``ReadTimeoutError``)
- a busy or briefly unavailable service: HTTP 408, 429, 500, 502, 503 or
  504, an FTP 4xx reply such as "421 too many users", or an FTP server that
  closed the connection
- input/output trouble that goes away: ``EIO``, a stale NFS handle, a busy
  or interrupted call

Anything else is 4: another HTTP status (a bucket in another region, a
403, a 404), an FTP 5xx reply, a file that is not there or may not be used,
or a full disk. PYSTILT decides from the error and what caused it, not from
its type, because s3fs and fsspec's HTTP filesystem report many permanent
answers as a plain ``OSError`` or ``FileNotFoundError``. Every ``stilt``
command follows this table, except that a closed standard output
(``stilt status | head``) ends quietly with 1, and ``stilt fetch`` gives 1
for a file the archive does not have.

The same rule applies to an error inside one simulation, such as a worker
that cannot write its result to the output bucket. An error a retry may get
past is not a failure of the simulation: no failure record is written (an
earlier attempt's is removed), the simulation stays ``pending``, or
``interrupted`` once its run had started, the other simulations go on, and
the run ends with 3. Any other error, a ``SimulationError`` such as a failed
HYSPLIT run included, is a failure record and exit 1. When both happen in
one run it exits 3, and the retry, which runs only what did not finish,
ends with 1 if the failure remains. A bucket that stays down keeps every
try at 3, so the program that drives the retries sets how many to make.

For a Kubernetes indexed Job, a ``podFailurePolicy`` that retries on 3 and
fails the task on 1 and 4 follows from this table.

When the Job is done, ``stilt status /project`` counts the finished and
failed simulations of the whole project. ``stilt status /project --json``
gives the same counts, per state and variant, as JSON for the program
that drives the Job.
