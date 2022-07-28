import os
import shutil

input_ws = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\landsat'
output_ws = r'Y:\justinh\Projects\SNWA\update_20161017\fmask_update'

for year in os.listdir(input_ws):
    input_year_ws = os.path.join(input_ws, str(year))
    output_year_ws = os.path.join(output_ws, str(year))
    if not os.path.isdir(input_year_ws):
        continue
    if not os.path.isdir(output_year_ws):
        os.makedirs(output_year_ws)

    for item in os.listdir(input_year_ws):
        if 'fmask.tif' in item:
            shutil.copy(
                os.path.join(input_year_ws, item),
                os.path.join(output_year_ws, item))
