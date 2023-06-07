# NALM For Publication

**run_LLM.py** fits yearly, periodically-varying logistic growth parameters to mosquito process-based model output

## Required user inputs:

**data_file:** path/name of data file

**output_path:** output path

**loc:** name of location ('Toronto')

**pop:** population type ('Total' or 'Active')

**start1:** initial start day of mosquito fitting season (MFS) in 'mm-dd' format ('05-01')

**start2:** final start day of MFS in 'mm-dd' format ('06-01')

**end1:** initial end day of MFS in 'mm-dd' format, for total population only ('10-01')

**end2:** final end of day MFS in 'mm-dd' format, for total population only ('11-01')

**start_year:** initial full year of data ('2005')

**end_year:** final year of data ('2020')

note that all inputs should be entered as strings

## Output files:

**All_fits.csv:** returns parameter fittings for all start/end day combinations for each year

**Best_fits.csv:** returns parameter fittings for the optimal start/end day combinations for each year


