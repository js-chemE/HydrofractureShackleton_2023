# Lake processes

## Main Steps

### As part of 01_raw
1. run GEE and export tiles with lake masks
2. run tiles_split2merged.py, in order to merge all the tiles that were splitted during the download from google drive

### As part of 02_processed
3. copy the entire folder to 02_processed
4. run tiles_delete_split_files.py to remove all the splitted files
5. run detect_drainges.py
6. run process_drainages.py
7. run plot_drainages_01_single.ipynb to plot all detected drainages, in folder 01_drainages_plots

8. manually inspect the detected drainages and place to true positive/interesting ones in folder 02_drainages_plots
9. run collect_true_drainages.py in order to collect based on the drainage_plots from 02_drainage_plots all drainages into one file
10. run plot_drainages_02_single.ipynb to plot for all remaining drainages L8/S2 and S1 series, exported into folder 03_drainages_plots

### As part of 03_cleaned
11. run manually_check_events.ipynb with added decissions/justifications to sort out remaining false positive, export drainages into 03_drainages
12. run read_out_values.ipynb in order to dmg, act, lai, fuerst values
13. run create_clean.ipynb in order to obtain clean .shp


## Side Steps

### Lake Extent
after main Step 4
1. run tiles_merge2extent.py to produce extents per tile, year, sat
2. run tiles_merge2extent_sat.py to produce extents per tile, year
3. run tiles_merge_extents.py to produce extents per year
4. run vectorize_extents.py to extrend .shp from .tif 
