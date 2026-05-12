#!/usr/bin/env Rscript
# Emit intermediate R-STILT calc_footprint tables for Python parity tests.
#
# Args:
#   1  particles_parquet
#   2  output_dir
#   3  xmn
#   4  xmx
#   5  ymn
#   6  ymx
#   7  xres
#   8  yres
#   9  smooth_factor
#  10  time_integrate
#  11  stilt_r_dir
#  12  projection (optional; default "+proj=longlat")

args <- commandArgs(trailingOnly = TRUE)
particles_parquet <- args[1]
output_dir        <- args[2]
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

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

options(dplyr.summarise.inform = FALSE)
src_dir <- file.path(stilt_r_dir, "r", "src")
for (f in list.files(src_dir, pattern = "\\.r$", full.names = TRUE)) {
  source(f)
}

load_libs('arrow', 'dplyr', 'proj4')

p <- as.data.frame(read_parquet(particles_parquet))

# Match calc_footprint.r wrapper behavior for minimal synthetic inputs.
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

np <- length(unique(p$indx))
time_sign <- sign(median(p$time))
is_longlat <- grepl("+proj=longlat", projection, fixed = TRUE)

if (is_longlat) {
  xdist <- ((180 - xmn) - (-180 - xmx)) %% 360
  if (xdist == 0) {
    xmn <- -180
    xmx <- 180
  } else if ((xmx < xmn) || (xmx > 180)) {
    p$long <- wrap_longitude_antimeridian(p$long)
    xmn <- wrap_longitude_antimeridian(xmn)
    xmx <- wrap_longitude_antimeridian(xmx)
  }
}

distances <- p %>%
  filter(abs(time) < 100) %>%
  group_by(indx) %>%
  summarize(dx = median(abs(diff(long))),
            dy = median(abs(diff(lati)))) %>%
  ungroup()

should_interpolate <- (median(distances$dx, na.rm = TRUE) > xres) ||
  (median(distances$dy, na.rm = TRUE) > yres)
if (should_interpolate) {
  times <- c(seq(0, 10, by = 0.1),
             seq(10.2, 20, by = 0.2),
             seq(20.5, 100, by = 0.5)) * time_sign

  aptime <- abs(p$time)
  foot_0_10_sum <- sum(p$foot[aptime <= 10], na.rm = TRUE)
  foot_10_20_sum <- sum(p$foot[aptime > 10 & aptime <= 20], na.rm = TRUE)
  foot_20_100_sum <- sum(p$foot[aptime > 20 & aptime <= 100], na.rm = TRUE)

  p <- p %>%
    full_join(expand.grid(time = times,
                          indx = unique(p$indx)),
              by = c("indx", "time")) %>%
    arrange(indx, -time) %>%
    group_by(indx) %>%
    mutate(long = na_interp(long, x = time),
           lati = na_interp(lati, x = time),
           foot = na_interp(foot, x = time)) %>%
    ungroup() %>%
    na.omit() %>%
    mutate(time = round(time, 1))

  aptime <- abs(p$time)
  mi <- aptime <= 10
  p$foot[mi] <- p$foot[mi] / (sum(p$foot[mi], na.rm = TRUE) / foot_0_10_sum)
  mi <- aptime > 10 & aptime <= 20
  p$foot[mi] <- p$foot[mi] / (sum(p$foot[mi], na.rm = TRUE) / foot_10_20_sum)
  mi <- aptime > 20 & aptime <= 100
  p$foot[mi] <- p$foot[mi] / (sum(p$foot[mi], na.rm = TRUE) / foot_20_100_sum)
}

write_parquet(p, file.path(output_dir, "p_after_interp.parquet"))

p <- p %>%
  group_by(indx) %>%
  mutate(rtime = time - (time_sign) * min(abs(time))) %>%
  ungroup()

if (!is_longlat) {
  require(proj4)
  p[, c("long", "lati")] <- project(p[, c("long", "lati")], projection)
  grid_lims <- project(list(c(xmn, xmx), c(ymn, ymx)), projection)
  xmn <- min(grid_lims$x)
  xmx <- max(grid_lims$x)
  ymn <- min(grid_lims$y)
  ymx <- max(grid_lims$y)
}

write_parquet(p, file.path(output_dir, "p_with_rtime.parquet"))

glong <- head(seq(xmn, xmx, by = xres), -1)
glati <- head(seq(ymn, ymx, by = yres), -1)
write_parquet(data.frame(axis = "x", value = glong),
              file.path(output_dir, "grid_x.parquet"))
write_parquet(data.frame(axis = "y", value = glati),
              file.path(output_dir, "grid_y.parquet"))

kernel <- p %>%
  group_by(rtime) %>%
  summarize(varsum = var(long) + var(lati),
            lati = mean(lati)) %>%
  ungroup() %>%
  na.omit()

di <- kernel$varsum^(1/4)
ti <- abs(kernel$rtime / 1440)^(1/2)
grid_conv <- if (is_longlat) cos(kernel$lati * pi / 180) else 1
kernel$w <- smooth_factor * 0.06 * di * ti / grid_conv
write_parquet(kernel, file.path(output_dir, "kernel.parquet"))

xyres <- c(xres, yres)
max_k <- make_gauss_kernel(xyres, max(kernel$w), projection)
xbuf <- ncol(max_k)
xbufh <- (xbuf - 1) / 2
ybuf <- nrow(max_k)
ybufh <- (ybuf - 1) / 2
glong_buf <- seq(xmn - (xbuf * xres), xmx + ((xbuf - 1) * xres), by = xres)
glati_buf <- seq(ymn - (ybuf * yres), ymx + ((ybuf - 1) * yres), by = yres)

p_grid <- p %>%
  filter(foot > 0,
         long >= (xmn - xbufh * xres), long < (xmx + xbufh * xres),
         lati >= (ymn - ybufh * yres), lati < (ymx + ybufh * yres)) %>%
  transmute(loi = as.integer(findInterval(long, glong_buf)),
            lai = as.integer(findInterval(lati, glati_buf)),
            foot = foot,
            time = time,
            rtime) %>%
  group_by(loi, lai, time, rtime) %>%
  summarize(foot = sum(foot, na.rm = TRUE)) %>%
  ungroup()

p_grid$layer <- if (time_integrate) 0 else floor(p_grid$time / 60)
write_parquet(p_grid, file.path(output_dir, "raster.parquet"))
write_parquet(data.frame(np = np), file.path(output_dir, "metadata.parquet"))
