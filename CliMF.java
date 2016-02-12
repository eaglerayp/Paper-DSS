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
//import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
//import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.AbstractFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
//import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

import java.util.*;

/** Matrix factorization with user and item biases for rating prediction, trained with plain vanilla SGD  */
public class CliMF extends AbstractFactorizer {
    private class Usersitemidset {
        private long[] replyidset;
        private long[] noreplyidset;

        public Usersitemidset() {
        }

        public long[] getReplyidset() {
            return replyidset;
        }

        public long[] getNoreplyidset() {
            return noreplyidset;
        }


        public void setReplyidset(long[] replyidset) {
            this.replyidset = (replyidset == null ? null : replyidset.clone());
        }

        public void setNoreplyidset(long[] noreplyidset) {
            this.noreplyidset = (noreplyidset == null ? null : noreplyidset.clone());
        }

    }

    protected static final int FEATURE_OFFSET = 2;
    /** place in user vector where the bias is stored */
    protected static final int USER_BIAS_INDEX = 0;
    /** place in item vector where the bias is stored */
    protected static final int ITEM_BIAS_INDEX = 1;
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
    protected final HashSet<Long> SocialIDset = new HashSet<Long>();
    protected long testuser;
    protected long coveruser;
    protected final HashMap<Long, Long> postwriter;
    public int left_records = 0;
    public int Total_Test_records = 0;
    public int Total_records = 0;

    public CliMF(DataModel dataModel, DataModel SocialdataModel, DataModel WriterdataModel,
                 int numFeatures, int numIterations, long testuser, long coveruser,
                 HashMap<Long, Long> postwriter) throws TasteException {
        this(dataModel, SocialdataModel, WriterdataModel, numFeatures, 1, 0.1, 0.01, numIterations, 0.95, testuser,
             coveruser, postwriter);
    }

    public CliMF(DataModel dataModel, DataModel SocialdataModel, DataModel WriterdataModel, int numFeatures,
                 double learningRate, double preventOverfitting,
                 double randomNoise, int numIterations, double learningRateDecay, long testuser, long coveruser,
                 HashMap<Long, Long> postwriter) throws TasteException {
        super(dataModel);
        this.dataModel = dataModel;
        this.SocialdataModel = SocialdataModel;
        this.WriterdataModel = WriterdataModel;
        this.numFeatures = numFeatures + FEATURE_OFFSET;
        this.numIterations = numIterations;
        this.learningRate = learningRate;
        this.learningRateDecay = learningRateDecay;
        this.preventOverfitting = preventOverfitting;
        this.randomNoise = randomNoise;
        this.testuser = testuser;
        this.coveruser = coveruser;
        this.postwriter = postwriter;
    }

    protected void prepareTraining() throws TasteException {
        RandomWrapper random = (RandomWrapper) RandomUtils.getRandom(0L);
        userVectors = new double[dataModel.getNumUsers()][numFeatures];
        itemVectors = new double[dataModel.getNumItems()][numFeatures];
        LongPrimitiveIterator socialuser = SocialdataModel.getUserIDs();
        while (socialuser
                .hasNext()) { //create social date model idset because user may have no friend ,and it will cause bug
            long userID = socialuser.nextLong();
            SocialIDset.add(userID);
        }
//        double globalAverage = getAveragePreference();
        for (int userIndex = 0; userIndex < userVectors.length; userIndex++) {
            userVectors[userIndex][USER_BIAS_INDEX] = 0; // will store user bias
            userVectors[userIndex][ITEM_BIAS_INDEX] = 1; // corresponding item feature contains item bias
            for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                userVectors[userIndex][feature] = random.nextGaussian() * randomNoise;
            }
            //unit vectorize
            userVectors[userIndex] = unitvectorize(userVectors[userIndex]);
        }
        for (int itemIndex = 0; itemIndex < itemVectors.length; itemIndex++) {
            itemVectors[itemIndex][USER_BIAS_INDEX] = 1; // corresponding user feature contains user bias
            itemVectors[itemIndex][ITEM_BIAS_INDEX] = 0; // will store item bias
            for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
                itemVectors[itemIndex][feature] = random.nextGaussian() * randomNoise;
            }
            //unit vectorize
            itemVectors[itemIndex] = unitvectorize(itemVectors[itemIndex]);
        }

        //compute bias
        try {
            computeBias();
        } catch (TasteException e) {
            System.out.println("error in bias computing");
        }
    }

    public void computeBias() throws TasteException {
        //user bias
        double userAverage = 0;
        double userMax = -Double.MAX_VALUE;
        double userMin = Double.MAX_VALUE;
        LongPrimitiveIterator userIDs = dataModel.getUserIDs();
        while (userIDs.hasNext()) {
            long userid = userIDs.nextLong();
            double user_sum = 0;
            for (Preference record : dataModel.getPreferencesFromUser(userid)) {
                user_sum += record.getValue();
            }
            userAverage += user_sum;
            userMax = (user_sum > userMax) ? user_sum : userMax;
            userMin = (user_sum < userMin) ? user_sum : userMin;
            int userindex = userIndex(userid);
            userVectors[userindex][USER_BIAS_INDEX] = user_sum;
        }
        userAverage /= dataModel.getNumUsers();
        double min_dist = userAverage - userMin;
        double max_dist = userMax - userAverage;
        double normalize = (min_dist > max_dist) ? min_dist : max_dist;
        userIDs = dataModel.getUserIDs();
        while (userIDs.hasNext()) {
            long userid = userIDs.nextLong();
            int userindex = userIndex(userid);
            userVectors[userindex][USER_BIAS_INDEX] -= userAverage;
            userVectors[userindex][USER_BIAS_INDEX] /= normalize;
        }

        //item bias
        double itemAverage = 0;
        double itemMax = -Double.MAX_VALUE;
        double itemMin = Double.MAX_VALUE;
        LongPrimitiveIterator itemIDs = dataModel.getItemIDs();
        while (itemIDs.hasNext()) {
            long itemid = itemIDs.nextLong();
            double item_sum = 0;
            for (Preference record : dataModel.getPreferencesForItem(itemid)) {
                item_sum += record.getValue();
            }
            itemAverage += item_sum;
            itemMax = (item_sum > itemMax) ? item_sum : itemMax;
            itemMin = (item_sum < itemMin) ? item_sum : itemMin;
            int itemindex = itemIndex(itemid);
            itemVectors[itemindex][ITEM_BIAS_INDEX] = item_sum;
        }
        itemAverage /= dataModel.getNumItems();
        min_dist = itemAverage - itemMin;
        max_dist = itemMax - itemAverage;
        normalize = (min_dist > max_dist) ? min_dist : max_dist;
        itemIDs = dataModel.getItemIDs();
        while (itemIDs.hasNext()) {
            long itemid = itemIDs.nextLong();
            int itemindex = itemIndex(itemid);
            itemVectors[itemindex][ITEM_BIAS_INDEX] -= itemAverage;
            itemVectors[itemindex][ITEM_BIAS_INDEX] /= normalize;
        }
    }

    @Override
    public Factorization factorize() throws TasteException {
        System.out.println("in CLIMF");
        prepareTraining();
        double currentLearningRate = learningRate;
        HashSet<Long> DMusers = new HashSet<Long>();
        LongPrimitiveIterator ud = dataModel.getUserIDs();
        while (ud.hasNext()) {
            DMusers.add(ud.nextLong());
        }
        for (int it = 0; it < numIterations; it++) {
            for (long userID : DMusers) {
                //以user出發去跑iteration
                updateParameters(userID, currentLearningRate);
            }
            currentLearningRate *= learningRateDecay;
        }//end learning iteration
        System.out.println("Test records:" + Total_Test_records / numIterations);
        //aggregate writer feature  produce a new user-write vectors replace itemVectors
        double[][] writerVectors = new double[WriterdataModel.getNumUsers()][numFeatures];
        LongPrimitiveIterator it = WriterdataModel.getUserIDs();
        while (it.hasNext()) {
            long userid = it.nextLong();
            int userIndex = userIndex(userid);
            for (int i = 0; i < numFeatures; i++) {
                writerVectors[userIndex][i] = 0.0;
            }
            int post_count = 0;
            for (long item : WriterdataModel.getItemIDsFromUser(userid)) {
                int itemIndex = itemIndex(item);
                if (itemIndex < dataModel.getNumItems()) {
                    post_count++;
                    for (int i = 0; i < numFeatures; i++) {
                        writerVectors[userIndex][i] += itemVectors[itemIndex][i];
                    }
                }
            }
            if (post_count > 0) {
                //unit vector
                writerVectors[userIndex] = unitvectorize(writerVectors[userIndex]);
                writerVectors[userIndex][USER_BIAS_INDEX] /= post_count;
                writerVectors[userIndex][ITEM_BIAS_INDEX] /= post_count;
            }
        }
        int n = (int) (Math.random() * dataModel.getNumUsers());
        //System.out.println("writer"+n+" user bias:"+writerVectors[n][USER_BIAS_INDEX]);
        // System.out.println("writer"+n+"item bias:"+writerVectors[n][ITEM_BIAS_INDEX]);
        assert writerVectors[n][USER_BIAS_INDEX] > 0.99;
        assert writerVectors[n][USER_BIAS_INDEX] < 1.01;

        return createFactorization(userVectors, writerVectors);//(userVectors, itemVectors);
    }


//    double getAveragePreference() throws TasteException {
//        RunningAverage average = new FullRunningAverage();
//        LongPrimitiveIterator it = dataModel.getUserIDs();
//        while (it.hasNext()) {
//            for (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
//                average.addDatum(pref.getValue());
//            }
//        }
//        return average.getAverage();
//    }

    protected void updateParameters(long userID, double currentLearningRate) throws TasteException {
        int userIndex = userIndex(userID);
        double[] userVector = userVectors[userIndex];

        Usersitemidset idset = new Usersitemidset();
        splitReplyIDs(userID, idset);
        long[] replyids = idset.getReplyidset();
        double[] g_fij = new double[replyids.length];

        double[] sum_sgdu_cof = new double[replyids.length];
        double sum_sgdv_cof = 0.0;
//        get g_fij
        for (int index_rj = 0; index_rj < replyids.length; index_rj++) {
            long replyid_j = replyids[index_rj];
            double fij = F_inner_product(userID, replyid_j);
            double g_negative_fij = logistic(-fij);
            g_fij[index_rj] = g_negative_fij;

            for (long replyid_k : replyids) {
                if (replyid_k != replyid_j) {
                    double fik = F_inner_product(userID, replyid_k);
                    double g_fik_fij = logistic(fik - fij);
                    double g_fij_fik = logistic(fij - fik);

                    double sgdu_item_son = g_fik_fij * g_fij_fik;
                    double sgdu_item_mother = 1.0 - g_fik_fij;
                    double sgdu_item = sgdu_item_son / sgdu_item_mother;

                    double sgdv_second_term = taylor_function(g_fik_fij) - taylor_function(g_fij_fik);
                    double sgdv_item = g_fij_fik * g_fik_fij * sgdv_second_term;
                    sum_sgdu_cof[index_rj] += sgdu_item;
                    sum_sgdv_cof += sgdv_item;
                }
            }
        }

//        update ui
        for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
            double pF = userVector[feature];
            double sgdu = 0.0;
            for (int index_rj = 0; index_rj < replyids.length; index_rj++) {
                long replyid_j = replyids[index_rj];
                int qjIndex = itemIndex(replyid_j);
                double Vj = itemVectors[qjIndex][feature];

                double sum_sgdu_second_item = 0.0;
                for (long replyid_k : replyids) {
                    int qkIndex = itemIndex(replyid_k);
                    double Vk = itemVectors[qkIndex][feature];
                    double Vjk = Vj - Vk;
                    sum_sgdu_second_item += Vjk * sum_sgdu_cof[index_rj];
                }

                sgdu += g_fij[index_rj] * Vj + sum_sgdu_second_item;
            }
            sgdu -= preventOverfitting * pF;

            userVector[feature] += currentLearningRate * sgdu;
//            update all relevant vj
            for (int index_rj = 0; index_rj < replyids.length; index_rj++) {
                long replyid_j = replyids[index_rj];
                int qjIndex = itemIndex(replyid_j);
                double[] itemVector = itemVectors[qjIndex];
                double sgdvj = g_fij[index_rj] + sum_sgdv_cof;
                sgdvj *= pF;
                sgdvj -= preventOverfitting * itemVector[feature];
                itemVector[feature] += sgdvj*currentLearningRate;
            }
        }
//        unitVectorize
        for (int index_rj = 0; index_rj < replyids.length; index_rj++) {
            long replyid_j = replyids[index_rj];
            int qjIndex = itemIndex(replyid_j);
            double[] itemVector = itemVectors[qjIndex];
            unitvectorize(itemVector);
        }

        unitvectorize(userVector);
    }

    private double taylor_function(double x) {
        double mother = 1 - x;
        return (double) 1 / mother;
    }

    private double F_inner_product(long userID, long itemID) {
        double piqu = 0;
        int itemindex = itemIndex(itemID);
        int userindex = userIndex(userID);

        //compute piqu
        for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {//  puqi
            piqu += userVectors[userindex][feature] * itemVectors[itemindex][feature];
        }

        assert piqu < 1.1;
        return piqu;
    }

    private double logistic(double x) {
        double one = 1.0;
        double mother = 1 + Math.pow(Math.E, -x);
        return one / mother;
    }

    private void splitReplyIDs(long userID, Usersitemidset idset) throws TasteException {
        FastIDSet itemids = dataModel.getItemIDsFromUser(userID);
        FastIDSet replyids = new FastIDSet();
        FastIDSet noreplyids = new FastIDSet();
        for (long item : itemids) {
            long writer = postwriter.get(item);
            if (dataModel.getPreferenceValue(userID, item) > 0) {
                if (testuser == userID && writer == coveruser) {//蓋掉testuser和coveruser information
                    Total_Test_records++;
                } else {
                    replyids.add(item);
                }
            } else {
                noreplyids.add(item);
            }
        }
        if (userID == testuser) {
            left_records += replyids.size();
        }
        Total_records += replyids.size();
        idset.setReplyidset(replyids.toArray());
        idset.setNoreplyidset(noreplyids.toArray());
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
        //prevent overflow
        double max = -Double.MAX_VALUE;
        for (int i = FEATURE_OFFSET; i < numFeatures; i++) {
            max = (Math.abs(vector[i]) > max) ? Math.abs(vector[i]) : max;
        }
        double max_square = max * max;
        for (int i = FEATURE_OFFSET; i < numFeatures; i++) {
            vectorlength += vector[i] * vector[i] / max_square;
        }
        vectorlength = max * Math.sqrt(vectorlength);
        return vectorlength;
    }
}
