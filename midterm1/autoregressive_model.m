% import the data and sort the data by date
appliances = readtable("energydata_complete.csv");
appliances = sortrows(appliances, "date");
appliances = appliances(:, "Appliances");
appliances = appliances{:,:};