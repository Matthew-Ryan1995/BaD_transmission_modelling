# This script to is take the parameters in `parameter_ranges.json` and write it out to `parameters_table.tex`, ready to input into LaTeX writeups as required. Requires latex packages \usepackage{pdflscape}  % for landscape tables
# \usepackage{afterpage}
import json
from astropy.io import ascii

# latex table file name
latex_file = '../tabs/parameter_table.tex'

# pull in parameter ranges from file
with open("parameter_ranges.json") as json_file:
    json_data = json.load(json_file)

# get the number of parameters from this
num_params = len(json_data)

description = []
units = []
source = []
location = []
symbols = []
param_mins = []
param_exps = []
param_maxs = []

counter = 0
for key, value in json_data.items():
    description.append(value["description"])
    units.append(value["units"])
    source.append(value["source"])
    location.append(value["location"])
    symbols.append(value["symbol"])
    param_mins.append(value["min"])
    param_exps.append(value["exp"])
    param_maxs.append(value["max"])

    counter = counter + 1

# now need to rearrange these dictionaries into numpy arrays
#data = {symbol, description, exp_value(min,max), units, source, location}

param_vals = []
for i in range(num_params):
    param_vals.append(
        str(param_exps[i]) + ' (' + str(param_mins[i]) + ', ' + str(param_maxs[i]) + ')')

data = [symbols, description, param_vals, units, source, location]
names = ['\\hline Symbol', 'Description',
         'Value(min,max)', 'Units', 'Source(s)', 'Source location']

# fiddle with the latex formatting
data_start_text = r'\endfirsthead' + '\n' \
                  r'\multicolumn{6}{c}' + '\n' \
                  r'{\tablename\ \thetable\ -- \textit{Continued from previous page}} \\' + '\n' \
                  r'\hline' + '\n' \
                  r'Symbol & Description & Value(min,max) & Units & Source(s) & Source location(s) \\' + '\n' \
                  r'\hline' + '\n' \
                  r'\endhead' + '\n' \
                  r'\hline \multicolumn{6}{r}{\textit{Continued on next page}} \\' + '\n' \
                  r'\endfoot' + '\n' \
                  r'\hline' + '\n' \
                  r'\endlastfoot' + '\n'

caption_text = r'\caption{Table of parameters.} \label{tab:parameters} \\' + '\n'
pretable_text = r'\afterpage{%' + '\n' \
    r'\clearpage% Flush earlier floats (otherwise order might not be correct)' + '\n' \
    r'\begin{landscape}% Landscape page' + '\n' \
    r'\begin{center}'
posttable_text = r'\end{center}' + '\n' \
                 r'\end{landscape}' + '\n' \
                 r'\clearpage% Flush page' + '\n' \
                 r'}'

latexchoices = {'tabletype': None,
                'col_align': '|l|p{3cm}|p{3cm}|l|p{2cm}|p{2cm}|', 'data_start': data_start_text}


# write data as a latex table, and out to file
ascii.write(data, format='latex', names=names,
            latexdict=latexchoices, output=latex_file, overwrite=True)

# # need to clean up the latex
# # i.e. open file, read out current contents
f = open(latex_file, 'r')
contents = f.read()
f.close()

# change the contents
# # add pretable_text to the beginning
# # change "tabular" to "longtable"
# # add post-table text to the end
new_contents = contents.replace('tabular', 'longtable')
new_contents = [pretable_text + '\n' + new_contents + '\n' + posttable_text]

# write the new contents back over the top
f = open(latex_file, 'w')
new_contents = "".join(new_contents)
f.write(new_contents)
f.close()


# read out line by line to insert caption and other useful lines
f = open(latex_file, 'r')
contents = f.readlines()
f.close()

# add caption NB: if add anything else need to update insertion linenumber
contents.insert(5, caption_text)

# calculate line numbers to add level categorisation
new_contents = "".join(contents)

# write latest refinements back over the top
f = open(latex_file, 'w')
f.write(new_contents)
f.close()


print("finished")
