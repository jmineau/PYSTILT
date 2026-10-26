#!/usr/bin/env Rscript
# Write a small STILT-R project with STILT-R's own code: out/by-id/ and the
# params.RDS rslurm keeps. It is the test sample of
# stilt.migration.read_stilt_r (tests/fixtures/stilt_r_project/), and the
# r_only test writes it again with the pinned STILT-R.
#
# Args (positional):
#   1  stilt_r_dir  A uataq/stilt checkout with r/src/permute.so built
#   2  project_dir  Project folder to write (made if missing); the
#                   simulations go in its out/by-id
#
# The particles are made up here: no HYSPLIT and no met. The rest is what
# STILT-R's simulation_step() does with them:
#   - read_config() reads a CONC.CFG into `params`;
#   - calc_plume_dilution() adds the near-field correction, which leaves a
#     tibble with foot_no_hnf_dilution;
#   - write_control() writes the CONTROL file HYSPLIT reads;
#   - saveRDS() writes the list (file, receptor, particle, params) as
#     <id>_traj.rds;
#   - calc_footprint() writes <id>_foot.nc, with smooth_factor 0.5.
# Then the rows stilt_apply() hands rslurm::slurm_apply() are saved as
# _rslurm_<project>/params.RDS, as rslurm saves them: one row per
# simulation, with the before_trajec and before_footprint functions and a
# few of the other arguments of simulation_step().
#
# Two simulations:
#   201507050000_-111.85_40.77_5   a point receptor, run backward 2 hours,
#                                  with hourly footprint layers and a grid
#                                  whose cells are not square.
#   201507051800_-111.9_40.7_X     a column receptor as X-STILT makes one (two
#                                  heights at one place, so particles have
#                                  xhgt), released at 18:00:30, with a
#                                  time-integrated footprint.
#
# output$file and the met files in CONTROL are written relative, so no local
# path is stored in the sample.

args <- commandArgs(trailingOnly = TRUE)
stilt_r_dir <- args[1]
project_dir <- args[2]
by_id_dir <- file.path(project_dir, 'out', 'by-id')

dyn.load(file.path(stilt_r_dir, 'r', 'src', 'permute.so'))
options(dplyr.summarise.inform = FALSE)
for (f in list.files(file.path(stilt_r_dir, 'r', 'src'), pattern = '\\.r$',
                     full.names = TRUE)) {
  source(f)
}
suppressPackageStartupMessages({
  library(dplyr)
  library(ncdf4)
  library(raster)
})

# A CONC.CFG as HYSPLIT writes it, shortened. read_config() turns '' into NA
# and keeps .FALSE. as a string, as it does for a real file.
conc_cfg <- c(" &SETUP", " tratio = 0.75,", " delt = 1.0,", " numpar =  10,",
              " maxpar = 10,", " efile = '',", " krand =  4,",
              " wvert = .FALSE.,", " kmix0 = 150,", " frvs = 0.01,", " /")
n_hours <- -2
numpar <- 10
# Not STILT-R's default of 1, so a test can tell the value read from
# params.RDS from a default. These particles spread so little that each
# kernel is one cell at 0.5 or 1, so the footprints are the same either way.
smooth_factor <- 0.5
met_files <- file.path('met', c('20150704_18-23_hrrr', '20150705_00-05_hrrr'))

make_particles <- function(numpar, n_minutes, long, lati, zagl, seed) {
  set.seed(seed)
  p <- expand.grid(indx = seq_len(numpar), time = -seq_len(n_minutes))
  p <- p[order(p$indx, -p$time), ]
  drift <- rnorm(numpar, 0.002, 0.0005)[p$indx]
  # Rounded as HYSPLIT writes PARTICLE_STILT.DAT, which also keeps the
  # files small.
  p$long <- round(long - drift * abs(p$time) + rnorm(nrow(p), 0, 0.0005), 4)
  p$lati <- round(lati + 0.0004 * abs(p$time) + rnorm(nrow(p), 0, 0.0005), 4)
  p$zagl <- round(pmax(1, zagl + 2 * abs(p$time) + rnorm(nrow(p), 0, 5)), 2)
  p$mlht <- 800
  p$dens <- round(1.0 + rnorm(nrow(p), 0, 0.01), 5)
  p$samt <- 1
  p$sigw <- 0.5
  p$tlgr <- 60
  p$foot <- 0.02897 / (0.5 * p$mlht * p$dens) * p$samt * 60
  p[, c('time', 'indx', 'long', 'lati', 'zagl', 'foot', 'mlht', 'dens',
        'samt', 'sigw', 'tlgr')]
}

write_simulation <- function(receptor, particle, xmn, xmx, ymn, ymx, xres,
                             yres, time_integrate) {
  zagl_id <- if (length(receptor$zagl) > 1) 'X' else receptor$zagl
  simulation_id <- strftime(
    receptor$run_time,
    paste0('%Y%m%d%H%M_', receptor$long, '_', receptor$lati, '_', zagl_id),
    'UTC')
  rundir <- file.path(by_id_dir, simulation_id)
  dir.create(rundir, recursive = TRUE, showWarnings = FALSE)

  write_control(receptor, emisshrs = 0.01, n_hour = n_hours,
                met_files = met_files, file = file.path(rundir, 'CONTROL'))

  cfg <- file.path(tempdir(), 'CONC.CFG')
  writeLines(conc_cfg, cfg)

  output <- list()
  output$file <- file.path('out', 'by-id', simulation_id,
                           paste0(simulation_id, '_traj.rds'))
  output$receptor <- receptor
  output$particle <- particle
  output$params <- read_config(cfg)
  saveRDS(output, file.path(rundir, paste0(simulation_id, '_traj.rds')))

  calc_footprint(output$particle[, c('time', 'indx', 'long', 'lati', 'foot')],
                 output = file.path(rundir, paste0(simulation_id, '_foot.nc')),
                 r_run_time = receptor$run_time,
                 projection = '+proj=longlat',
                 smooth_factor = smooth_factor, time_integrate = time_integrate,
                 xmn = xmn, xmx = xmx, xres = xres,
                 ymn = ymn, ymx = ymx, yres = yres)
}

# A point receptor, backward 2 hours, hourly layers.
point <- list(run_time = as.POSIXct('2015-07-05 00:00:00', tz = 'UTC'),
              lati = 40.77, long = -111.85, zagl = 5)
p <- make_particles(numpar, 120, point$long, point$lati, point$zagl, seed = 1)
p <- calc_plume_dilution(p, numpar, point$zagl, 0.5)
write_simulation(point, p, xmn = -112.2, xmx = -111.6, ymn = 40.6, ymx = 41.0,
                 xres = 0.02, yres = 0.025, time_integrate = FALSE)

# A column receptor, as X-STILT writes one: release heights kept as xhgt
# (calc_trajectory's formula), then the near-field correction.
column <- list(run_time = as.POSIXct('2015-07-05 18:00:30', tz = 'UTC'),
               lati = 40.7, long = -111.9, zagl = c(0, 1000))
p <- make_particles(numpar, 90, column$long, column$lati, 500, seed = 2)
step <- (max(column$zagl) - min(column$zagl)) / numpar
px <- data.frame(indx = 1:numpar)
px$xhgt <- (px$indx - 0.5) * step + min(column$zagl)
p <- merge(p, px, by = 'indx', sort = FALSE)
p <- calc_plume_dilution(p, numpar, column$zagl, 0.5)
write_simulation(column, p, xmn = -112.2, xmx = -111.6, ymn = 40.6,
                 ymx = 41.0, xres = 0.02, yres = 0.02, time_integrate = TRUE)

# What rslurm saves: the hooks as list columns of functions, as STILT-R's
# run_stilt.r passes them, the receptors, and the settings, one row each.
before_trajec <- function() output
before_footprint <- function() output
params <- tibble(
  simulation_id = NA,
  before_footprint = list(before_footprint),
  before_trajec = list(before_trajec),
  n_hours = n_hours,
  numpar = numpar,
  r_run_time = c(point$run_time, column$run_time),
  r_lati = c(point$lati, column$lati),
  r_long = c(point$long, column$long),
  r_zagl = list(point$zagl, column$zagl),
  smooth_factor = smooth_factor,
  time_integrate = c(FALSE, TRUE)
)
rslurm_dir <- file.path(project_dir, paste0('_rslurm_', basename(project_dir)))
dir.create(rslurm_dir, showWarnings = FALSE)
saveRDS(params, file.path(rslurm_dir, 'params.RDS'))
