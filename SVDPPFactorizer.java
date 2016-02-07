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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.common.RandomWrapper;

import java.util.*;

/**
 * SVD++, an enhancement of classical matrix factorization for rating prediction.
 * Additionally to using ratings (how did people rate?) for learning, this model also takes into account
 * who rated what.
 *
 * Yehuda Koren: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model, KDD 2008.
 * http://research.yahoo.com/files/kdd08koren.pdf
 */
public final class SVDPPFactorizer extends RatingSGDFactorizer {
    private double[][] p; // user NN
    private double[][] y; // item NN
    private Map<Integer, List<Integer>> itemsByUser;
    private final int numIterations;

    public SVDPPFactorizer(DataModel dataModel, DataModel SocialdataModel, DataModel WriterdataModel,
                           int numFeatures, int numIterations, long testuser, long coveruser,
                           HashMap<Long, Long> postwriter) throws TasteException {
        this(dataModel, SocialdataModel, WriterdataModel, numFeatures, 0.7, 0.1, 0.01, numIterations, 0.95, testuser,
             coveruser, postwriter);
    }

    public SVDPPFactorizer(DataModel dataModel, DataModel SocialdataModel, DataModel WriterdataModel,
                           int numFeatures, double learningRate, double preventOverfitting,
                           double randomNoise, int numIterations, double learningRateDecay, long testuser,
                           long coveruser, HashMap<Long, Long> postwriter) throws TasteException {
        super(dataModel, SocialdataModel, WriterdataModel, numFeatures, learningRate, preventOverfitting, randomNoise,
              numIterations, learningRateDecay, testuser, coveruser, postwriter);
        this.test_record_count = 0;
        this.numIterations = numIterations;
    }

    //first 20,770,40,1,33
    protected void prepareTraining() throws TasteException {
        super.prepareTraining();
        Random random = RandomUtils.getRandom(10L);
        p = new double[dataModel.getNumUsers()][numFeatures];
        for (int i = 0; i < p.length; i++) {
            for (int feature = 0; feature < FEATURE_OFFSET; feature++) {
                p[i][feature] = 0;
            }
            for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                p[i][feature] = random.nextGaussian() * randomNoise;
            }
        }

        y = new double[dataModel.getNumItems()][numFeatures];
        for (int i = 0; i < y.length; i++) {
            for (int feature = 0; feature < FEATURE_OFFSET; feature++) {
                y[i][feature] = 0;
            }
            for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                y[i][feature] = random.nextGaussian() * randomNoise;
            }
        }

    /* get internal item IDs which we will need several times */
        itemsByUser = Maps.newHashMap();
        LongPrimitiveIterator userIDs = dataModel.getUserIDs();
        while (userIDs.hasNext()) {
            long userId = userIDs.nextLong();
            int userIndex = userIndex(userId);
            FastIDSet itemIDsFromUser = dataModel.getItemIDsFromUser(userId);
            List<Integer> itemIndexes = Lists.newArrayListWithCapacity(itemIDsFromUser.size());
            itemsByUser.put(userIndex, itemIndexes);
//             cover testuser's reply items
            if (userId == testuser) {
                for (long itemID2 : itemIDsFromUser) {
                    if (coveruser != (postwriter.get(itemID2))) {
                        int i2 = itemIndex(itemID2);
                        itemIndexes.add(i2);
                    }
                }
            } else {
                for (long itemID2 : itemIDsFromUser) {
                    int i2 = itemIndex(itemID2);
                    itemIndexes.add(i2);
                }
            }
        }

        LongPrimitiveIterator socialuser = SocialdataModel.getUserIDs();
        while (socialuser
                .hasNext()) { //create social date model idset because user may have no friend ,and it will cause bug
            long userID = socialuser.nextLong();
            SocialIDset.add(userID);
        }
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
                    if (testuser == userId) {
                        if (coveruser == (postwriter.get(itemId))) {
                            test_record_count++;
                        } else {
                            updateParameters(userId, itemId, rating, currentLearningRate);
                        }
                    } else {
                        updateParameters(userId, itemId, rating, currentLearningRate);
                    }
                } else {
                    updateParameters(userId, itemId, rating, currentLearningRate);
                }
            }
            currentLearningRate *= learningRateDecay;
        }

        //svd++'s update
        for (int userIndex = 0; userIndex < userVectors.length; userIndex++) {
            for (int itemIndex : itemsByUser.get(userIndex)) {
                for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                    userVectors[userIndex][feature] += y[itemIndex][feature];
                }
            }
            double denominator = Math.sqrt(itemsByUser.get(userIndex).size());
            for (int feature = 0; feature < userVectors[userIndex].length; feature++) {
                userVectors[userIndex][feature] =
                        (float) (userVectors[userIndex][feature] / denominator + p[userIndex][feature]);
            }
        }

        System.out.println("Test records:" + test_record_count / numIterations);
        //aggregate writer feature  produce a new user-write vectors replace itemVectors
        double[][] writerVectors = new double[WriterdataModel.getNumUsers()][numFeatures];
        LongPrimitiveIterator it = WriterdataModel.getUserIDs();
        while (it.hasNext()) {
            long userid = it.nextLong();
            int userIndex = userIndex(userid);
            for (int i = 0; i < numFeatures; i++) {
                writerVectors[userIndex][i] = 0.0;
            }
            int write = 0;
            for (long item : WriterdataModel.getItemIDsFromUser(userid)) {
                int itemIndex = itemIndex(item);
                if (itemIndex < dataModel.getNumItems()) {
                    write++;
                    for (int i = 0; i < numFeatures; i++) {
                        writerVectors[userIndex][i] += itemVectors[itemIndex][i];
                    }
                }
            }
            if (write > 0) {
                writerVectors[userIndex] = unitvectorize(writerVectors[userIndex]);
                writerVectors[userIndex][USER_BIAS_INDEX] /= write;
                writerVectors[userIndex][ITEM_BIAS_INDEX] /= write;
            }
        }
        return createFactorization(userVectors, writerVectors);
    }

    protected void updateParameters(long userID, long itemID, float rating, double currentLearningRate) {
        int userIndex = userIndex(userID);
        int itemIndex = itemIndex(itemID);

        double[] userVector = p[userIndex];
        double[] itemVector = itemVectors[itemIndex];

        double[] pPlusY = new double[numFeatures];
        for (int i2 : itemsByUser.get(userIndex)) {
            for (int f = FEATURE_OFFSET; f < numFeatures; f++) {
                pPlusY[f] += y[i2][f];
            }
        }
        double denominator = Math.sqrt(itemsByUser.get(userIndex).size());
        for (int feature = 0; feature < pPlusY.length; feature++) {
            pPlusY[feature] = (float) (pPlusY[feature] / denominator + p[userIndex][feature]);
        }

        double prediction = predictRating(pPlusY, itemIndex);
        double err = rating - prediction;
        double normalized_error = err / denominator;

//        // adjust user bias
//        userVector[USER_BIAS_INDEX] +=
//                biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * userVector[USER_BIAS_INDEX]);
//
//        // adjust item bias
//        itemVector[ITEM_BIAS_INDEX] +=
//                biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * itemVector[ITEM_BIAS_INDEX]);

        // adjust features
        for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
            double pF = userVector[feature];
            double iF = itemVector[feature];

            double deltaU = err * iF - preventOverfitting * pF;
            userVector[feature] += currentLearningRate * deltaU;

            double deltaI = err * pPlusY[feature] - preventOverfitting * iF;
            itemVector[feature] += currentLearningRate * deltaI;

            double commonUpdate = normalized_error * iF;
            for (int itemIndex2 : itemsByUser.get(userIndex)) {
                double deltaI2 = commonUpdate - preventOverfitting * y[itemIndex2][feature];
                y[itemIndex2][feature] += learningRate * deltaI2;
            }
        }
        unitvectorize(itemVector);
        unitvectorize(userVector);
        for (int itemIndex2 : itemsByUser.get(userIndex)) {
            unitvectorize(y[itemIndex2]);
        }
    }

    private double predictRating(double[] userVector, int itemID) {
        double sum = 0;
        for (int feature = 0; feature < numFeatures; feature++) {
            sum += userVector[feature] * itemVectors[itemID][feature];
        }
        return sum;
    }

    public double[] unitvectorize(double[] vector) {
        double vectorlength = vectorlength(vector);
        for (int i = FEATURE_OFFSET; i < numFeatures; i++) {
            vector[i] /= vectorlength;
        }
        return vector;
    }

    public double vectorlength(double[] vector) {
        double vectorlength = 0.0;
        for (int i = FEATURE_OFFSET; i < numFeatures; i++) {
            vectorlength += vector[i] * vector[i];
        }
        vectorlength = Math.sqrt(vectorlength);
        return vectorlength;
    }
}
