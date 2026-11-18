#!/usr/bin/env Rscript
# Write the SETUP.CFG that STILT-R's run_stilt.r gives HYSPLIT, without a run.
#
# Args:
#   1  stilt_r_dir
#   2  output path, ending in SETUP.CFG
#
# Every top-level assignment in run_stilt.r whose name is an argument of
# write_setup() (numpar, varsiwant, capemin, ...) is evaluated in order, and
# STILT-R's own write_setup() writes the file from them. Nothing else in
# run_stilt.r runs.

args <- commandArgs(trailingOnly = TRUE)
stilt_r_dir <- args[1]
output <- args[2]

source(file.path(stilt_r_dir, "r", "src", "write_setup.r"))
wanted <- setdiff(names(formals(write_setup)), c("file", "..."))

values <- new.env()
for (expr in parse(file.path(stilt_r_dir, "r", "run_stilt.r"))) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      is.name(expr[[2]]) && as.character(expr[[2]]) %in% wanted) {
    assign(as.character(expr[[2]]), eval(expr[[3]], values), envir = values)
  }
}

invisible(do.call(write_setup, c(as.list(values), file = output)))
