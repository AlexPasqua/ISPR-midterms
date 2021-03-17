% import the data and sort the data by date
appliances = readtable("energydata_complete.csv");
appliances = sortrows(appliances, "date");
appliances = appliances(:, "Appliances");
appliances = appliances{:,:};

% plot(appliances)
[r,lg] = xcorr(appliances);
[a,e,k] = levinson(r, 2);

pred = appliances(1) * a(2) + appliances(2) * a(3);
disp("Pred: "); disp(pred);
disp("Actual: "); disp(appliances(3));