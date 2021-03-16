% import the data
appliances = readtable("energydata_complete.csv");
appliances = appliances(:, "Appliances");
appliances = appliances{:,:};