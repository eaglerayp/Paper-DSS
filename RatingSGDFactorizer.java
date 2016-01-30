/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.predictionmarketing.itemrecommend;


import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.AbstractFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

import java.util.*;

/** Matrix factorization with user and item biases for rating prediction, trained with plain vanilla SGD  */
public class RatingSGDFactorizer extends AbstractFactorizer {

    protected static final int FEATURE_OFFSET = 2;

    /** Multiplicative decay factor for learning_rate */
    protected final double learningRateDecay;
    /** Learning rate (step size) */
    protected final double learningRate;
    /** Parameter used to prevent overfitting. */
    protected final double preventOverfitting;
    /** Number of features used to compute this factorization */
    protected final int numFeatures;
    /** Number of iterations */
    private final int numIterations;
    /** Standard deviation for random initialization of features */
    protected final double randomNoise;
    /** User features */
    protected double[][] userVectors;
    /** Item features */
    protected double[][] itemVectors;
    protected final DataModel dataModel;
    protected final DataModel SocialdataModel;
    protected final DataModel WriterdataModel;
    protected long[] cachedUserIDs;
    protected long[] cachedItemIDs;
    protected final HashSet<Long> SocialIDset=new HashSet<Long>();
    protected double biasLearningRate = 0.1;
    protected double biasReg = 0.1;

    /** place in user vector where the bias is stored */
    protected static final int USER_BIAS_INDEX = 0;
    /** place in item vector where the bias is stored */
    protected static final int ITEM_BIAS_INDEX = 1;
    protected final HashMap<Long,Long> postwriter;
    public  int test_record_count;
    protected final long testuser;
    protected final long coveruser;

    static <K,V extends Comparable<? super V>>
    SortedSet<Map.Entry<K,V>> entriesSortedByValues(Map<K,V> map) {
        SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(
                new Comparator<Map.Entry<K,V>>() {
                    @Override public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
                        int res = e2.getValue().compareTo(e1.getValue());
                        return res != 0 ? res : 1;
                    }
                }
        );
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    public RatingSGDFactorizer(DataModel dataModel, DataModel SocialdataModel, DataModel WriterdataModel,
                               int numFeatures, int numIterations, long testuser, long coveruser,
                               HashMap<Long, Long> postwriter) throws TasteException {
        this(dataModel,SocialdataModel,WriterdataModel, numFeatures, 0.01, 0.1, 0.01, numIterations, 0.95, testuser,coveruser,postwriter);
    }

    public RatingSGDFactorizer(DataModel dataModel, DataModel SocialdataModel, DataModel WriterdataModel,
                               int numFeatures, double learningRate,
                               double preventOverfitting,
                               double randomNoise, int numIterations, double learningRateDecay,
                               long testuser, long coveruser, HashMap<Long, Long> postwriter) throws TasteException {
        super(dataModel);
        this.dataModel = dataModel;
        this.SocialdataModel=SocialdataModel;
        this.WriterdataModel = WriterdataModel;
        this.numFeatures = numFeatures + FEATURE_OFFSET;
        this.numIterations = numIterations;

        this.learningRate = learningRate;
        this.learningRateDecay = learningRateDecay;
        this.preventOverfitting = preventOverfitting;
        this.randomNoise = randomNoise;
        this.testuser=testuser;
        this.coveruser=coveruser;
        this.postwriter=postwriter;
        this.test_record_count=0;
    }

    protected void prepareTraining() throws TasteException {
        RandomWrapper random = (RandomWrapper) RandomUtils.getRandom(0L);
        userVectors = new double[dataModel.getNumUsers()][numFeatures];
        itemVectors = new double[dataModel.getNumItems()][numFeatures];
        LongPrimitiveIterator socialuser=SocialdataModel.getUserIDs();
        while (socialuser.hasNext()) { //create social date model idset because user may have no friend ,and it will cause bug
            long userID = socialuser.nextLong();
            SocialIDset.add(userID);
        }
        double globalAverage = getAveragePreference();
        for (int userIndex = 0; userIndex < userVectors.length; userIndex++) {
            userVectors[userIndex][0] = globalAverage;
            userVectors[userIndex][USER_BIAS_INDEX] = 0; // will store user bias
            userVectors[userIndex][ITEM_BIAS_INDEX] = 1; // corresponding item feature contains item bias
            for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                userVectors[userIndex][feature] = random.nextGaussian() * randomNoise;
            }
        }
        for (int itemIndex = 0; itemIndex < itemVectors.length; itemIndex++) {
            itemVectors[itemIndex][0] = 1; // corresponding user feature contains global average
            itemVectors[itemIndex][USER_BIAS_INDEX] = 1; // corresponding user feature contains user bias
            itemVectors[itemIndex][ITEM_BIAS_INDEX] = 0; // will store item bias
            for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                itemVectors[itemIndex][feature] = random.nextGaussian() * randomNoise;
            }
        }
//compute bias
        //user bias
        double userAverage=0;
        double userMax=-Double.MAX_VALUE;
        double userMin=Double.MAX_VALUE;
        LongPrimitiveIterator userIDs = dataModel.getUserIDs();
        while (userIDs.hasNext()) {
            long userid=userIDs.nextLong();
            double user_sum=0;
            for(Preference record : dataModel.getPreferencesFromUser(userid)){
                user_sum+=record.getValue();
            }
            userAverage+=user_sum;
            userMax=(user_sum>userMax)?user_sum:userMax;
            userMin=(user_sum<userMin)?user_sum:userMin;
            int userindex=userIndex(userid);
            userVectors[userindex][USER_BIAS_INDEX]=user_sum;
        }
        userAverage/=dataModel.getNumUsers();
        double min_dist=userAverage-userMin;
        double max_dist=userMax-userAverage;
        double normalize=(min_dist>max_dist)?min_dist:max_dist;
        userIDs = dataModel.getUserIDs();
        while (userIDs.hasNext()) {
            long userid=userIDs.nextLong();
            int userindex=userIndex(userid);
            userVectors[userindex][USER_BIAS_INDEX]-=userAverage;
            userVectors[userindex][USER_BIAS_INDEX]/=normalize;
        }
        //item bias
        double itemAverage=0;
        double itemMax=-Double.MAX_VALUE;
        double itemMin=Double.MAX_VALUE;
        LongPrimitiveIterator itemIDs = dataModel.getItemIDs();
        while (itemIDs.hasNext()) {
            long itemid=itemIDs.nextLong();
            double item_sum=0;
            for(Preference record : dataModel.getPreferencesForItem(itemid)){
                item_sum+=record.getValue();
            }
            itemAverage+=item_sum;
            itemMax=(item_sum>itemMax)?item_sum:itemMax;
            itemMin=(item_sum<itemMin)?item_sum:itemMin;
            int itemindex=itemIndex(itemid);
            itemVectors[itemindex][ITEM_BIAS_INDEX]=item_sum;
        }
        itemAverage/=dataModel.getNumItems();
        min_dist=itemAverage-itemMin;
        max_dist=itemMax-itemAverage;
        normalize=(min_dist>max_dist)?min_dist:max_dist;
        itemIDs = dataModel.getItemIDs();
        while (itemIDs.hasNext()) {
            long itemid=itemIDs.nextLong();
            int itemindex=itemIndex(itemid);
            itemVectors[itemindex][ITEM_BIAS_INDEX]-=itemAverage;
            itemVectors[itemindex][ITEM_BIAS_INDEX]/=normalize;
        }

        cachePreferences();
        shufflePreferences();
    }

    private int countPreferences() throws TasteException {
        int numPreferences = 0;
        LongPrimitiveIterator userIDs = dataModel.getUserIDs();
        while (userIDs.hasNext()) {
            PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userIDs.nextLong());
            numPreferences += preferencesFromUser.length();
        }
        return numPreferences;
    }

    private void cachePreferences() throws TasteException {
        int numPreferences = countPreferences();
        cachedUserIDs = new long[numPreferences];
        cachedItemIDs = new long[numPreferences];

        LongPrimitiveIterator userIDs = dataModel.getUserIDs();
        int index = 0;
        while (userIDs.hasNext()) {
            long userID = userIDs.nextLong();
            PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userID);
            for (Preference preference : preferencesFromUser) {
                cachedUserIDs[index] = userID;
                cachedItemIDs[index] = preference.getItemID();
                index++;
            }
        }
    }

    protected void shufflePreferences() {
        RandomWrapper random = RandomUtils.getRandom();
    /* Durstenfeld shuffle */
        for (int currentPos = cachedUserIDs.length - 1; currentPos > 0; currentPos--) {
            int swapPos = random.nextInt(currentPos + 1);
            swapCachedPreferences(currentPos, swapPos);
        }
    }

    private void swapCachedPreferences(int posA, int posB) {
        long tmpUserIndex = cachedUserIDs[posA];
        long tmpItemIndex = cachedItemIDs[posA];

        cachedUserIDs[posA] = cachedUserIDs[posB];
        cachedItemIDs[posA] = cachedItemIDs[posB];

        cachedUserIDs[posB] = tmpUserIndex;
        cachedItemIDs[posB] = tmpItemIndex;
    }

    @Override
    public Factorization factorize() throws TasteException {
        prepareTraining();

        double currentLearningRate = learningRate;


        for (int it = 0; it < numIterations; it++) {
            for (int index = 0; index < cachedUserIDs.length; index++) {
                long userId = cachedUserIDs[index];
                long itemId = cachedItemIDs[index];
                float rating = dataModel.getPreferenceValue(userId, itemId);
                if (rating > 0) {//蓋掉testuser和coveruser 的 reply record
                    if(testuser==userId) {
                        if(coveruser==(postwriter.get(itemId))){
                            test_record_count++;
                        }else{
                            updateParameters(userId, itemId, rating, currentLearningRate);
                        }
                    }else{
                        updateParameters(userId, itemId, rating, currentLearningRate);
                    }
                }else{
                    updateParameters(userId, itemId, rating, currentLearningRate);
                }
            }
            currentLearningRate *= learningRateDecay;
        }
        System.out.println("Test records:"+test_record_count/numIterations);
        //aggregate writer feature  produce a new user-write vectors replace itemVectors
        double[][] writerVectors= new double[WriterdataModel.getNumUsers()][numFeatures];
        LongPrimitiveIterator it = WriterdataModel.getUserIDs() ;
        while (it.hasNext()) {
            long userid=it.nextLong();
            int userIndex = userIndex(userid);
            for(int i=0;i<numFeatures;i++){
                writerVectors[userIndex][i]=0.0;
            }
            int write=0;
            for (long item : WriterdataModel.getItemIDsFromUser(userid)) {
                int itemIndex=itemIndex(item);
                if(itemIndex< dataModel.getNumItems()) {
                    write++;
                    for (int i = 0; i < numFeatures; i++) {
                        writerVectors[userIndex][i] += itemVectors[itemIndex][i];
                    }
                }
            }
            if(write>0) {
                writerVectors[userIndex] = unitvectorize(writerVectors[userIndex]);
                writerVectors[userIndex][USER_BIAS_INDEX]/=write;
                writerVectors[userIndex][ITEM_BIAS_INDEX]/=write;
            }
        }
        return createFactorization(userVectors, writerVectors);
    }
    public Factorization weighted_factorize() throws TasteException {
        //aggregate aggregated writer feature  produce a new user-write vectors replace itemVectors
        double[][] writerVectors= new double[WriterdataModel.getNumUsers()][numFeatures];
        LongPrimitiveIterator it = WriterdataModel.getUserIDs() ;
        while (it.hasNext()) {
            long userid=it.nextLong();
            int userIndex = userIndex(userid);
            for(int i=0;i<numFeatures;i++){
                writerVectors[userIndex][i]=0.0;
            }
            FastIDSet updateset=  WriterdataModel.getItemIDsFromUser(userid);
            int post_count=updateset.size();
            for (long item : updateset) {
                int itemIndex=itemIndex(item);
                if(itemIndex< dataModel.getNumItems()) {
                    double[] weight_array=new double[numFeatures];
                    HashMap<Integer,Double> itemvaluemap=new HashMap<Integer, Double>();
                    for (int i = 0; i < numFeatures; i++) {   //find max feature
                        itemvaluemap.put(i, itemVectors[itemIndex][i]);
                        weight_array[i]=1;
                    }
                    if(post_count>1) {
                        SortedSet<Map.Entry<Integer, Double>> sorted_map = entriesSortedByValues(itemvaluemap);
                        int i=0;
                        for (Map.Entry<Integer, Double> entry : sorted_map) {
                            weight_array[entry.getKey()] += 1;
                            i++;
                            if(i>0){
                                break;
                            }
                        }
                    }
                    for (int i = 0; i < numFeatures; i++) {   //aggregate
                        writerVectors[userIndex][i] =weight_array[i]*itemVectors[itemIndex][i];
                    }
                }
            }
            if(post_count>0) {
                //unit vector
                writerVectors[userIndex] = unitvectorize(writerVectors[userIndex]);
                writerVectors[userIndex][USER_BIAS_INDEX]/=post_count;
                writerVectors[userIndex][ITEM_BIAS_INDEX]/=post_count;
            }
        }
        return createFactorization(userVectors,writerVectors);
    }
    double getAveragePreference() throws TasteException {
        RunningAverage average = new FullRunningAverage();
        LongPrimitiveIterator it = dataModel.getUserIDs();
        while (it.hasNext()) {
            for (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
                average.addDatum(pref.getValue());
            }
        }
        return average.getAverage();
    }

    protected void updateParameters(long userID, long itemID, float rating, double currentLearningRate) {
        int userIndex = userIndex(userID);
        int itemIndex = itemIndex(itemID);

        double[] userVector = userVectors[userIndex];
        double[] itemVector = itemVectors[itemIndex];
        double prediction = predictRating(userIndex, itemIndex);
        double err = rating - prediction;
/*
        // adjust user bias
        userVector[USER_BIAS_INDEX] +=
                biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * userVector[USER_BIAS_INDEX]);

        // adjust item bias
        itemVector[ITEM_BIAS_INDEX] +=
                biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * itemVector[ITEM_BIAS_INDEX]);*/

        // adjust features
        for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
            double userFeature = userVector[feature];
            double itemFeature = itemVector[feature];

            double deltaUserFeature = err * itemFeature - preventOverfitting * userFeature;
            userVector[feature] += currentLearningRate * deltaUserFeature;

            double deltaItemFeature = err * userFeature - preventOverfitting * itemFeature;
            itemVector[feature] += currentLearningRate * deltaItemFeature;
        }
        itemVectors[itemIndex]=unitvectorize(itemVector);
        userVectors[userIndex]=unitvectorize(userVector);
    }

    private double predictRating(int userID, int itemID) {
        double sum = 0;
        for (int feature = 0; feature < numFeatures; feature++) {
            sum += userVectors[userID][feature] * itemVectors[itemID][feature];
        }
        return sum;
    }
    public double[] unitvectorize(double []vector){
        double vectorlength=vectorlength(vector);
        for(int i=FEATURE_OFFSET;i<numFeatures;i++){
            vector[i]/=vectorlength;
        }
        return vector;
    }
    public double vectorlength(double[] vector){
        double vectorlength=0.0;
        for(int i=FEATURE_OFFSET;i<numFeatures;i++){
            vectorlength+= vector[i]*vector[i];
        }
        vectorlength=Math.sqrt(vectorlength);
        return vectorlength;
    }
}
