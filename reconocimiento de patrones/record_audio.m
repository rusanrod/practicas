recObject = audiorecorder(16000,16,1,-1);
recDuration = 1.5;
disp("Begin Speaking")
recordblocking(recObject,recDuration);
disp("End of recording")

paso = getaudiodata(recObject);
play(recObject);

plot(paso)