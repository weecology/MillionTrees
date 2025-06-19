### Download NEON tower plots and clip to tile extent
library(batchtools)
library(TreeSegmentation)
library(neonUtilities)
library(dplyr)

# Add a try-catch block to handle existing registry
tryCatch({
  reg <- loadRegistry(file.dir = "/home/b.weinstein/logs/process_neon_plots/", writeable = TRUE)
  print("Registry loaded")
  clearRegistry()
}, error = function(e) {
  reg <- makeRegistry(file.dir = "/home/b.weinstein/logs/process_neon_plots/")
  print("Registry created")
})

setwd("/home/b.weinstein/TreeSegmentation/analysis")

reg$cluster.functions <- makeClusterFunctionsSlurm(
  template = "detection_template.tmpl",
  array.jobs = TRUE,
  nodename = "localhost",
  scheduler.latency = 5,
  fs.latency = 65
)
process_site <- function(site, year) {
  TreeSegmentation::crop_rgb_plots(site, year = year)
  
  tryCatch({
    TreeSegmentation::crop_lidar_plots(site, year = year)
  }, error = function(e) {
    print(paste("Error processing lidar for site:", site))
  })
  
  tryCatch({
    TreeSegmentation::crop_CHM_plots(site, year)
  }, error = function(e) {
    print(paste("Error processing CHM for site:", site))
  })
}

sites <- c("ABBY", "ARIK", "BARR", "BART", "BLAN", "BONA", "CLBJ", "CPER", "CUPE", "DEJU", "DELA", "DSNY", "GRSM", "GUAN",
           "GUIL", "HARV", "HEAL", "HOPB", "JERC", "JORN", "KONZ", "LAJA", "LENO", "LIRO", "MCDI", "MLBS", "MOAB", "NIWO", "NOGP", "OAES", "OSBS", "PRIN", "PUUM", "REDB", "RMNP", "SCBI", "SERC", "SJER", "SOAP", "SRER", "STEI", "STER", "TALL", "TEAK", "TOOL", "UKFS", "UNDE", "WLOU", "WOOD", "WREF", "YELL")


# for (year in c(2018, 2019, 2020, 2021, 2022, 2023, 2024)) {
#   for (site in sites) {
#     process_site(site, year = as.character(year))
#   }
# }

# Create a data frame with all combinations of sites and years
site_year_combinations <- expand.grid(
  site = sites,
  year = c("2018", "2019", "2020", "2021", "2022", "2023", "2024"),
  stringsAsFactors = FALSE
)

# Create jobs for each combination
ids <- batchMap(fun = process_site, site = site_year_combinations$site, year = site_year_combinations$year)

# Run in chunks of 1
ids[, chunk := chunk(job.id, chunk.size = 20)]

# Set resources: enable memory measurement
res <- list(measure.memory = TRUE, walltime = "12:00:00", memory = "7GB")

# Submit jobs using the currently configured cluster functions
submitJobs(ids, resources = res, reg = reg)
waitForJobs(ids, reg = reg)
getStatus(reg = reg)
getErrorMessages(ids, missing.as.error = TRUE, reg = reg)
print(getJobTable())
