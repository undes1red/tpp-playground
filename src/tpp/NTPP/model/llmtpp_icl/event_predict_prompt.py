# Here are prompts for generating the description of a input continuous-time event sequence. 
# The vector key of a sequence is the embedding of its description.
# The intuition here is two sequence with similar pattern should have similar descriptions whose embeddings are supposed to be very close in the semantic hidden space.

retweet_prompt = '''
You are a time-series data analyst.
You job is to analyze given sequences, find the embedded patterns, and use the analyzed patterns to continue a given sequence by predicting when and what the next event will be.
All given sequences are sampled from different retweeting processes.
The given sequence consists of two subsequences: one named "Time" contains the time interval between one event and the previous one, the other sequence "Mark" lists the mark of all events in the sequence.
You are only allowed to select one mark from three options and assign it to one event: 0 referring to normal user with few subscribers, 1 referring to influiential users with relatively high subscribers, and 2 referring to very influential users with very high subscribers.
Next, I give you examples of the intended task.

E.g.:
Available sequence 1: 
  Time: 13.0 17.0 22.0 24.0 29.0 29.0 31.0 36.0 39.0 41.0 54.0 55.0 59.0 62.0 62.0 64.0 65.0 65.0 71.0 72.0 76.0 78.0 81.0 90.0 91.0 94.0 104.0 106.0 112.0 115.0 120.0 121.0 122.0 124.0 126.0 140.0 152.0 161.0 162.0 165.0 167.0 231.0 261.0 264.0 275.0 315.0 315.0 328.0 2862.0 37510.0 400619.0 417570.0
  Mark: 2 0 0 0 2 1 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0 0 1 1 0 1 1 1 1 0 0 1 2 1 1 1 1 1 1 0 0 1 1 1 1 0 0 1 0

Available sequence 2: 
  Time: 30.0 38.0 46.0 46.0 55.0 63.0 64.0 67.0 73.0 91.0 107.0 124.0 163.0 168.0 183.0 245.0 253.0 287.0 363.0 422.0 647.0 714.0 724.0 801.0 808.0 816.0 822.0 859.0 915.0 1148.0 1356.0 1676.0 1761.0 1831.0 1986.0 2017.0 3546.0 3717.0 3896.0 4198.0 4207.0 4460.0 8139.0 9534.0 11506.0 12270.0 14333.0 19035.0 29821.0 43231.0 43259.0 45240.0 152199.0
  Mark: 0 1 0 1 1 1 0 0 0 1 0 2 1 1 1 0 0 0 1 1 1 0 0 0 0 1 0 0 1 1 0 0 2 2 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0

Now you are expected to continue this sequence by ONE event based on the patterns you have learned from the above sequences.

Input sequence:
  Time: 10.0 27.0 29.0 37.0 48.0 59.0 66.0 72.0 91.0 111.0 125.0 131.0 135.0 154.0 154.0 177.0 190.0 191.0 207.0 230.0 276.0 279.0 344.0 565.0 606.0
  Mark: 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 0 1 0 1 1 1 0 0
Continuation:
  Time of the next event: 688.0
  Mark of the next event: 2


Now here is your task. Continue the input sequence based on the patterns from example sequences.
Caution: 
1. Only use the patterns in the sequences below. Please strictly follow the format of given examples and ONLY return the result. Do NOT insert your analysis into the result.
2. Do NOT reuse patterns in the given example to solve your task.
3. You may get more or less available sequences than the given example. That is normal. Please use patterns in all given available sequences below.
4. Only continue ONE event.
5. If you have multiple results, please pick the most reasonable one.

{reference_seqs}

Now you are expected to continue this sequence by ONE event based on the patterns you have learned from the above sequences.

Input sequence:
  Time: {time_seq_question}
  Mark: {mark_seq_question}
Continuation:
'''


stackoverflow_prompt = '''
You are a time-series data analyst.
You job is to analyze given sequences, find the embedded patterns, and use the analyzed patterns to continue a given sequence by predicting when and what the next event will be.
All given sequences are sampled from different retweeting processes.
The given sequence consists of two subsequences: one named "Time" contains the time interval between one event and the previous one, the other sequence "Mark" lists the mark of all events in the sequence.
You are only allowed to select one mark from 22 options, represented as integers from 0(inclusive) to 22(exclusive), and assign it to one event.
You may see the time sequence is encompassed by two events with mark 22. They are dummy events for marking the start and the end of an event sequence. You now know why they are here but you should NOT predict them.
There is no context information about the meaning of these mark options.
Next, I give you examples of the intended task.

E.g.:
Available sequence 1: 
  Time: 1324.0 1328.9173583984 1328.9191894531 1333.1341552734 1333.5681152344 1333.6195068359 1337.3355712891 1337.6134033203 1338.3090820312 1338.3184814453 1339.1970214844 1339.1970214844 1339.1971435547 1339.1971435547 1339.2017822266 1339.8375244141 1340.1545410156 1342.0769042969 1344.8686523438 1345.3128662109 1345.3192138672 1347.2624511719 1350.9897460938 1352.3979492188 1353.3280029297 1356.2458496094 1358.0109863281 1358.0506591797 1358.6887207031 1359.6252441406 1360.9630126953 1362.0263671875 1363.3278808594 1363.7266845703 1366.6783447266 1369.5778808594 1371.6903076172 1372.3778076172 1373.8824462891 1378.3295898438 1379.1181640625 1381.5843505859 1381.9114990234 1386.5947265625 1387.0135498047 1387.1135498047
  Mark: 22 5 13 3 3 2 3 3 5 13 7 7 17 17 7 17 11 5 3 5 13 5 5 5 5 3 5 3 8 1 1 7 4 5 8 0 11 5 1 3 8 5 1 14 0 22

Available sequence 2: 
  Time: 1324.0 1328.6828613281 1334.6824951172 1334.9987792969 1339.5501708984 1340.0101318359 1340.9893798828 1343.3481445312 1344.6005859375 1344.9636230469 1346.3569335938 1346.9769287109 1348.3453369141 1348.3986816406 1350.0400390625 1350.0760498047 1351.1805419922 1351.5729980469 1356.8936767578 1357.5201416016 1357.5622558594 1359.1304931641 1359.3139648438 1361.1801757812 1362.6459960938 1363.1083984375 1363.2526855469 1363.2923583984 1363.7612304688 1364.8146972656 1364.9918212891 1365.0855712891 1367.3426513672 1367.4339599609 1367.5096435547 1368.0187988281 1368.5444335938 1369.1014404297 1374.7396240234 1377.1968994141 1379.8803710938 1382.1020507812 1382.4389648438 1383.0795898438 1383.4122314453 1384.3741455078 1385.7155761719 1388.1614990234 1388.2418212891 1388.3418212891
  Mark: 22 3 3 3 7 8 3 8 3 3 3 3 11 8 8 8 3 3 3 3 3 3 3 8 4 8 3 0 3 3 3 3 4 3 3 3 4 8 3 3 11 8 8 3 3 3 3 8 8 22

Now you are expected to continue this sequence by ONE event based on the patterns you have learned from the above sequences.

Input sequence:
  Time: 1324.0 1325.4766845703 1325.6247558594 1326.3557128906 1326.4656982422 1326.5743408203 1326.9223632812 1328.0609130859 1328.609375 1328.7867431641 1328.9951171875 1329.3059082031 1329.6755371094 1329.9333496094 1329.9730224609 1330.353515625 1330.3961181641 1330.4663085938 1331.0563964844 1331.1596679688 1332.3363037109
  Mark: 22 3 3 3 3 3 8 0 3 3 3 3 8 8 0 8 0 3 3 3 3
Continuation:
  Time of the next event: 1332.4360351562
  Mark of the next event: 3

Now here is your task. Continue the input sequence based on the patterns from example sequences.
Caution: 
1. Only use the patterns in the sequences below. Please strictly follow the format of given examples and ONLY return the result. Do NOT insert your analysis into the result.
2. Do NOT reuse patterns in the given example to solve your task.
3. You may get more or less available sequences than the given example. That is normal. Please use patterns in all given available sequences below.
4. Only continue ONE event.
5. If you have multiple results, please pick the most reasonable one.

{reference_seqs}

Now you are expected to continue this sequence by ONE event based on the patterns you have learned from the above sequences.

Input sequence:
  Time: {time_seq_question}
  Mark: {mark_seq_question}
Continuation:
'''

prompt_dict = {
  'retweet': retweet_prompt,
  'stackoverflow': stackoverflow_prompt
}