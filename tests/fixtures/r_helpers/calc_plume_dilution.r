#!/usr/bin/env Rscript
# Call R-STILT's calc_plume_dilution on a synthetic particle parquet.
#
# Args (positional):
#   1  particles_parquet  Input particle DataFrame (parquet)
#   2  output_parquet     Output particle DataFrame with corrected foot (parquet)
#   3  numpar             Number of particles (passed to R function; unused in formula)
#   4  r_zagl             Receptor height above ground (metres)
#   5  veght              STILT veght parameter (default 0.5)
#   6  stilt_r_dir        Path to uataq/stilt checkout

args <- commandArgs(trailingOnly = TRUE)
particles_parquet <- args[1]
output_parquet    <- args[2]
numpar            <- as.numeric(args[3])
r_zagl            <- as.numeric(args[4])
veght             <- as.numeric(args[5])
stilt_r_dir       <- args[6]

source(file.path(stilt_r_dir, "r", "src", "calc_plume_dilution.r"))

library(arrow)
library(dplyr)

p      <- as.data.frame(read_parquet(particles_parquet))
result <- calc_plume_dilution(p, numpar = numpar, r_zagl = r_zagl, veght = veght)
write_parquet(as.data.frame(result), output_parquet)
