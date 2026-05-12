#!/usr/bin/env Rscript
# Call R-STILT's calc_footprint on a synthetic particle parquet.
#
# Args (positional):
#   1  particles_parquet  Input particle DataFrame (parquet)
#   2  output_nc          Destination footprint NetCDF path
#   3  xmn                Grid xmin (degrees longitude)
#   4  xmx                Grid xmax
#   5  ymn                Grid ymin (degrees latitude)
#   6  ymx                Grid ymax
#   7  xres               Cell width (projection units)
#   8  yres               Cell height (projection units)
#   9  smooth_factor      Gaussian smoothing scale (1.0 = default)
#  10  time_integrate     "TRUE" or "FALSE"
#  11  stilt_r_dir        Path to uataq/stilt checkout
#  12  projection         Optional PROJ string (default "+proj=longlat")

args <- commandArgs(trailingOnly = TRUE)
particles_parquet <- args[1]
output_nc         <- args[2]
xmn               <- as.numeric(args[3])
xmx               <- as.numeric(args[4])
ymn               <- as.numeric(args[5])
ymx               <- as.numeric(args[6])
xres              <- as.numeric(args[7])
yres              <- as.numeric(args[8])
smooth_factor     <- as.numeric(args[9])
time_integrate    <- toupper(args[10]) == "TRUE"
stilt_r_dir       <- args[11]
projection        <- if (length(args) >= 12) args[12] else "+proj=longlat"

# Load the Fortran permute shared library that calc_footprint requires.
permute_so <- file.path(stilt_r_dir, "r", "src", "permute.so")
if (!file.exists(permute_so)) stop("permute.so not found: ", permute_so)
dyn.load(permute_so)

# Source all R helper functions from r/src/ (find_neighbor, na_interp, etc.)
# Suppress dplyr summarise grouping messages.
options(dplyr.summarise.inform = FALSE)
src_dir <- file.path(stilt_r_dir, "r", "src")
for (f in list.files(src_dir, pattern = "\\.r$", full.names = TRUE)) {
  source(f)
}

load_libs('arrow', 'dplyr', 'lubridate', 'ncdf4', 'raster', 'R.utils')
p <- as.data.frame(read_parquet(particles_parquet))

# Synthetic test DataFrames sometimes contain only one early-time row per
# particle, which causes two problems inside calc_footprint:
#
#  1. should_interpolate:  R filters |time| < 100 min and calls diff() per
#     indx.  If only one row passes the filter, diff() is empty, median = NA,
#     and the if (should_interpolate) conditional crashes.
#
#  2. kernel variance: var() of a single value is NA.  After na.omit() the
#     kernel DataFrame is empty, so max(w) = -Inf and make_gauss_kernel crashes.
#
# Fix: only for under-specified synthetic inputs, ensure every indx has at
# least two rows inside the first 100 minutes by adding t = 0 and t = -1 anchor
# rows (foot = 0, same position as the first real observation). Do not add
# anchors to real trajectory tables: zero-foot t = 0 rows alter rtime and
# therefore alter Gaussian smoothing bandwidths.
early_counts <- p %>%
  filter(abs(time) < 100) %>%
  count(indx, name = "n_early")
need_anchors <- nrow(early_counts) < length(unique(p$indx)) ||
  any(early_counts$n_early < 2)
if (need_anchors) {
  prototype <- p %>%
    group_by(indx) %>%
    slice(1) %>%
    ungroup() %>%
    mutate(foot = 0)
  t0  <- prototype %>% mutate(time =  0)
  tm1 <- prototype %>% mutate(time = -1)
  extra <- bind_rows(t0, tm1)
  extra <- extra[, names(p), drop = FALSE]
  p <- bind_rows(p, extra) %>% arrange(indx, -time)
}

# r_run_time determines the time-coordinate values in the output NetCDF.
# Synthetic tests compare only lat/lon/foot values, so any valid POSIXct works.
r_run_time <- as.POSIXct("2015-12-10 00:00:00", tz = "UTC")

# Bounds always arrive in lon/lat (degrees); R-STILT's calc_footprint expects
# the same convention and projects internally when projection != longlat.
calc_footprint(p,
               r_run_time     = r_run_time,
               output         = output_nc,
               xmn            = xmn,
               xmx            = xmx,
               ymn            = ymn,
               ymx            = ymx,
               xres           = xres,
               yres           = yres,
               smooth_factor  = smooth_factor,
               time_integrate = time_integrate,
               projection     = projection)
# When all particles are outside the domain, calc_footprint returns NULL and
# writes no file.  The Python test helper detects the missing file and builds
# an all-zero Dataset, so no action needed here.
