#!/usr/bin/env Rscript
# Run R-STILT calc_trajectory for one receptor and write the particle table.
#
# Args:
#   1  output_parquet
#   2  work_dir
#   3  stilt_r_dir
#   4  met_dir
#   5  met_file_format
#   6  met_file_tres
#   7  run_time ISO, UTC
#   8  receptor longitudes, comma separated
#   9  receptor latitudes, comma separated
#  10  receptor heights AGL, comma separated
#  11  n_hours
#  12  numpar
#  13  krand
#  14  seed
#  15  hnf_plume
#  16  rm_dat

args <- commandArgs(trailingOnly = TRUE)
output_parquet  <- args[1]
work_dir        <- args[2]
stilt_r_dir     <- args[3]
met_dir         <- args[4]
met_file_format <- args[5]
met_file_tres   <- args[6]
run_time        <- as.POSIXct(args[7], tz = "UTC", format = "%Y-%m-%dT%H:%M:%S")
long            <- as.numeric(strsplit(args[8], ",", fixed = TRUE)[[1]])
lati            <- as.numeric(strsplit(args[9], ",", fixed = TRUE)[[1]])
zagl            <- as.numeric(strsplit(args[10], ",", fixed = TRUE)[[1]])
n_hours         <- as.numeric(args[11])
numpar          <- as.numeric(args[12])
krand           <- as.numeric(args[13])
seed            <- as.numeric(args[14])
hnf_plume       <- toupper(args[15]) == "TRUE"
rm_dat          <- toupper(args[16]) == "TRUE"

dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_parquet), recursive = TRUE, showWarnings = FALSE)

options(dplyr.summarise.inform = FALSE)
src_dir <- file.path(stilt_r_dir, "r", "src")
for (f in list.files(src_dir, pattern = "\\.r$", full.names = TRUE)) {
  source(f)
}

library(arrow)

# Upstream R-STILT write_setup.r does not expose SEED.  PYSTILT does, and the
# live fidelity tests pin krand/seed for deterministic HYSPLIT output, so wrap
# R's writer and add SEED immediately before $END.
write_setup_orig <- write_setup
write_setup <- function(..., file = "SETUP.CFG") {
  out <- write_setup_orig(..., file = file)
  txt <- readLines(out)
  txt <- base::append(txt, paste0("SEED=", format(seed, scientific = FALSE), ","), after = length(txt) - 1)
  writeLines(txt, out)
  out
}

link_files(file.path(stilt_r_dir, "exe"), work_dir)
met_files <- find_met_files(run_time, n_hours, met_dir, met_file_format, met_file_tres)
if (length(met_files) < 1) {
  stop("No meteorology files found for R-STILT trajectory run.")
}

varsiwant <- c("time", "indx", "long", "lati", "zagl", "foot", "mlht", "dens",
               "samt", "sigw", "tlgr")
namelist <- list(
  varsiwant = varsiwant,
  capemin = -1,
  cmass = 0,
  conage = 48,
  cpack = 1,
  delt = 1,
  dxf = 1,
  dyf = 1,
  dzf = 0.01,
  efile = "",
  frhmax = 3,
  frhs = 1,
  frme = 0.1,
  frmr = 0,
  frts = 0.1,
  frvs = 0.01,
  hscale = 10800,
  ichem = 8,
  idsp = 2,
  initd = 0,
  k10m = 1,
  kagl = 1,
  kbls = 1,
  kblt = 5,
  kdef = 0,
  khinp = 0,
  khmax = 9999,
  kmix0 = 150,
  kmixd = 3,
  kmsl = 0,
  kpuff = 0,
  krand = krand,
  krnd = 6,
  kspl = 1,
  kwet = 1,
  kzmix = 0,
  maxdim = 1,
  maxpar = numpar,
  mgmin = 10,
  mhrs = 9999,
  nbptyp = 1,
  ncycl = 0,
  ndump = 0,
  ninit = 1,
  nstr = 0,
  nturb = 0,
  numpar = numpar,
  nver = 0,
  outdt = 0,
  p10f = 1,
  pinbc = "",
  pinpf = "",
  poutf = "",
  qcycle = 0,
  rhb = 80,
  rht = 60,
  splitf = 1,
  tkerd = 0.18,
  tkern = 0.18,
  tlfrac = 0.1,
  tout = 0,
  tratio = 0.75,
  tvmix = 1,
  veght = 0.5,
  vscale = 200,
  vscaleu = 200,
  vscales = -1,
  wbbh = 0,
  wbwf = 0,
  wbwr = 0,
  winderrtf = 0,
  wvert = FALSE,
  zicontroltf = 0,
  ziscale = rep(1, abs(n_hours))
)

output <- list(
  receptor = list(
    run_time = run_time,
    lati = lati,
    long = long,
    zagl = zagl
  )
)

particle <- calc_trajectory(
  namelist = namelist,
  rundir = work_dir,
  emisshrs = 0.01,
  hnf_plume = hnf_plume,
  met_files = met_files,
  n_hours = n_hours,
  output = output,
  rm_dat = rm_dat,
  timeout = 3600,
  w_option = 0,
  z_top = 25000
)
if (is.null(particle)) {
  stop("R-STILT calc_trajectory returned NULL.")
}

write_parquet(particle, output_parquet)
