package com.predictionmarketing.itemrecommend;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by eagle on 2015/1/15.
 * reviewed fro, 2016/02/09
 */
public class OtherMethod {
    public static HashMap<Long, Long> postwriter = new HashMap<Long, Long>();
    public static HashMap<Long, FastIDSet> SocialIDs = new HashMap<Long, FastIDSet>();
    public static HashMap<Long, HashSet<Long>> followerSet = new HashMap<Long, HashSet<Long>>();
    public static SortedSet<Map.Entry<Long, Integer>> sorted_map;
    public static int test_record_count = 0;
    protected static final int replyee_effectivelimit = 5;
    protected static final int friend_effectivelimit = 5;

    static <K, V extends Comparable<? super V>>
    SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<Map.Entry<K, V>>(
                new Comparator<Map.Entry<K, V>>() {
                    @Override
                    public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
                        int res = e2.getValue().compareTo(e1.getValue());
                        return res != 0 ? res : 1;
                    }
                }
        );
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    public static long[] similarityRecommend(DataModel dm, long userID, long coverid, int size,
                                             FastIDSet excludeids) throws
                                                                   TasteException {
        long[] result = new long[size];
        double[] maxscore = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = 0;
            maxscore[i] = -Double.MAX_VALUE;
        }
        FastIDSet useritem = dm.getItemIDsFromUser(userID);
        HashSet<Long> replyids = new HashSet<Long>();
        for (long item : useritem) {
            if (dm.getPreferenceValue(userID, item) > 0) {
                //蓋test資訊
                long writer = postwriter.get(item);
                if (coverid == writer) {
                    test_record_count++;
                } else {
                    replyids.add(item);
                }
            }
        }
        LongPrimitiveIterator dmit = dm.getUserIDs();
        while (dmit.hasNext()) {
            long compareuser = dmit.nextLong();
            //compute Jaccard similarity
            FastIDSet useritem2 = dm.getItemIDsFromUser(compareuser);
            HashSet<Long> replyids2 = new HashSet<Long>();
            for (long item : useritem2) {
//                long writer = postwriter.get(item);
                if (dm.getPreferenceValue(compareuser, item) > 0) {
                    replyids2.add(item);
                }
            }
            //intersection
            int intersection = 0;
            for (long item : replyids) {
                if (replyids2.contains(item)) {
                    intersection++;
                }
            }
            int union = replyids.size() + replyids2.size() - intersection;
            double score = (double) intersection / union;

            //find topn score
            if (compareuser != userID && !excludeids.contains(compareuser) && score > maxscore[size - 1]) {
                //change max and swap
                maxscore[size - 1] = score;
                result[size - 1] = compareuser;
                for (int z = size - 1; z > 0 && maxscore[z - 1] < maxscore[z]; z--) {
                    double tempscore = maxscore[z];
                    long tempindex = result[z];
                    maxscore[z] = maxscore[z - 1];
                    result[z] = result[z - 1];
                    maxscore[z - 1] = tempscore;
                    result[z - 1] = tempindex;
                }
            }
        }
        return result;
    }

    public static long[] friendoffriendRecommend(long userID, int size) throws
                                                                        TasteException {
        long[] result = new long[size];
        double[] maxscore = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = 0;
            maxscore[i] = -Double.MAX_VALUE;
        }
        HashMap<Long, Integer> FofFsize = new HashMap<Long, Integer>();
        FastIDSet friendid = new FastIDSet();
        if (SocialIDs.containsKey(userID)) {
            friendid = SocialIDs.get(userID);
        }
        for (long friend : friendid) {
            if (SocialIDs.containsKey(friend)) {
                for (long FofFid : SocialIDs.get(friend)) {
                    if (FofFsize.containsKey(FofFid)) {
                        int newsize = FofFsize.get(FofFid) + 1;
                        FofFsize.put(FofFid, newsize);
                    } else {
                        FofFsize.put(FofFid, 1);
                    }
                }
            }
        }

        //find topn 大
        for (long FofFid : FofFsize.keySet()) {
            double score = (double) FofFsize.get(FofFid);
            //normalized by his followsers size
            score /= followerSet.get(FofFid).size();
            if (score > maxscore[size - 1]) {
                //change max and swap
                maxscore[size - 1] = score;
                result[size - 1] = FofFid;
                for (int z = size - 1; z > 0 && maxscore[z - 1] < maxscore[z]; z--) {
                    double tempscore = maxscore[z];
                    long tempindex = result[z];
                    maxscore[z] = maxscore[z - 1];
                    result[z] = result[z - 1];
                    maxscore[z - 1] = tempscore;
                    result[z - 1] = tempindex;
                }
            }
        }
        /*int i =0;
        for(long FofFid:FofFsize.keySet()) {
            if(i<size) {
                result[i] = FofFid;
                i++;
            }
        }*/
        return result;
    }

    public static long[] followeroffriendRecommend(long userID, long coveruser, int size) throws
                                                                                          TasteException {
        long[] result = new long[size];
        double[] maxscore = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = 0;
            maxscore[i] = -Double.MAX_VALUE;
        }
        HashMap<Long, Integer> FofFsize = new HashMap<Long, Integer>();
        FastIDSet friendid = new FastIDSet();
        if (SocialIDs.containsKey(userID)) {
            friendid = SocialIDs.get(userID);
        }
        for (long friend : friendid) {
            if (followerSet.containsKey(friend)) {
                for (long FofFid : followerSet.get(friend)) {
                    if (FofFsize.containsKey(FofFid)) {
                        int newsize = FofFsize.get(FofFid) + 1;
                        FofFsize.put(FofFid, newsize);
                    } else {
                        FofFsize.put(FofFid, 1);
                    }
                }
            }
        }

        //find topn 大
        for (long FofFid : FofFsize.keySet()) {
            double score = (double) FofFsize.get(FofFid);
            //normalized by his followee's number
            score /= SocialIDs.get(FofFid).size();
            if (score > maxscore[size - 1]) {
                //change max and swap
                maxscore[size - 1] = score;
                result[size - 1] = FofFid;
                for (int z = size - 1; z > 0 && maxscore[z - 1] < maxscore[z]; z--) {
                    double tempscore = maxscore[z];
                    long tempindex = result[z];
                    maxscore[z] = maxscore[z - 1];
                    result[z] = result[z - 1];
                    maxscore[z - 1] = tempscore;
                    result[z - 1] = tempindex;
                }
            }
        }
        /*int i =0;
        for(long FofFid:FofFsize.keySet()) {
            if(i<size) {
                result[i] = FofFid;
                i++;
            }
        }*/
        return result;
    }

    public static long[] PopRec(FastIDSet excludeids, int size) {
        long result[] = new long[size];

        if (sorted_map.size() != size) {
            System.out.println(size);
            System.out.println(sorted_map.size());
        }
        int count = 0;
        for (Map.Entry<Long, Integer> entry : sorted_map) {
            long userid = entry.getKey();
            if (!excludeids.contains(userid)) {
                result[count++] = userid;
            }
        }
        return result;
    }

    public static class task {
        public long testuser;
        public long coveruser;
        public FastIDSet excludeids;
        public int interactions;
        public int totalinteractions;
        public int rank = 9999;

        public task(long testuser, long coveruser, FastIDSet excludeids, int interactions, int totalinteractions) {
            this.testuser = testuser;
            this.coveruser = coveruser;
            this.excludeids = excludeids;
            this.interactions = interactions;
            this.totalinteractions = totalinteractions;
        }

        public long getTestuser() {
            return testuser;
        }

        public long getCoveruser() {
            return coveruser;
        }

        public int getInteractions() {
            return interactions;
        }

        public FastIDSet getExcludeids() {
            return excludeids;
        }

        public int getTotalinteractions() {
            return totalinteractions;
        }

        public void setRank(int rank) {
            this.rank = rank;
        }

        public int getRank() {
            return rank;
        }
    }

    public static void main(String[] args) throws Exception {

        double starttime, endtime;
        starttime = System.currentTimeMillis();


        Writer outputwriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream("testdata-FdFd.csv"), "utf-8"));
        //import data,  using movielens for practicing
        outputwriter.write("testuser,coveruser,rank,interactions,totalinteractions" + "\n");

        DataModel dm = new FileDataModel(new File("data/replyscn300.csv"));
        DataModel writerdm = new FileDataModel(new File("data/writerscn300.csv"));
        DataModel socialdm = new FileDataModel(new File("data/socialscn300.csv"));

        int dm_reply_size = 0;
        int dm_noreply_size = 0;
        LongPrimitiveIterator dit = dm.getUserIDs();
        while (dit.hasNext()) {
            long userid = dit.nextLong();
            for (long itemid : dm.getItemIDsFromUser(userid)) {
                if (dm.getPreferenceValue(userid, itemid) > 0) {
                    dm_reply_size++;
                } else {
                    dm_noreply_size++;
                }
            }
        }
        System.out.println("dm_reply" + dm_reply_size);
        System.out.println("dm_noreply" + dm_noreply_size);


        LongPrimitiveIterator sit = socialdm.getItemIDs();
        while (sit.hasNext()) {
            long followeeid = sit.nextLong();
            HashSet<Long> followeridset = new HashSet<Long>();
            LongPrimitiveIterator suit = socialdm.getUserIDs();
            while (suit.hasNext()) {
                long userid = suit.nextLong();
                if (socialdm.getItemIDsFromUser(userid).contains(followeeid)) {
                    followeridset.add(userid);
                }
            }
            followerSet.put(followeeid, followeridset);
        }
        HashMap<Long, Integer> FriendsNum = new HashMap<Long, Integer>();
        for (long userid : followerSet.keySet()) {
            FriendsNum.put(userid, followerSet.get(userid).size());
        }
        sorted_map = entriesSortedByValues(FriendsNum);

        HashSet<Long> SocialIDset = new HashSet<Long>();
        LongPrimitiveIterator socialuser = socialdm.getUserIDs();
        while (socialuser
                .hasNext()) { //create social date model idset because user may have no friend ,and it will cause bug
            long userID = socialuser.nextLong();
            SocialIDset.add(userID);
        }
        for (Long userid : SocialIDset) {
            FastIDSet firendids = socialdm.getItemIDsFromUser(userid);
            SocialIDs.put(userid, firendids);
        }

        //construct post-writer map
        LongPrimitiveIterator it = writerdm.getUserIDs();
        while (it.hasNext()) {
            long userid = it.nextLong();
            for (long item : writerdm.getItemIDsFromUser(userid)) {
                postwriter.put(item, userid);
            }
        }

        System.out.println("start trainging");


        ArrayList<task> tasklist = generate_task(SocialIDset, socialdm, dm, postwriter);
        System.out.println("Task generated end, there are " + tasklist.size() + " tasks");

        //test
        ExecutorService executors = Executors.newFixedThreadPool(2);
        for (task taskk : tasklist) {
            executors.submit(new userOperate(dm, taskk));
        }
        executors.shutdown();
        try {
            executors.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException ee) {
            ee.printStackTrace();
        }

        //experiment end

        //start output result
        StatisticAndOutput(tasklist,outputwriter);
        outputwriter.close();
        endtime = System.currentTimeMillis();
        System.out.println((endtime - starttime) / 1000 + "SEC");
    }

    static class userOperate implements Runnable {
        private DataModel dm;
        private task atask;

        public userOperate(DataModel dm, task atask) {
            this.dm = dm;
            this.atask = atask;
        }

        public void run() {
            long testuser = atask.getTestuser();
            long coveruser = atask.getCoveruser();
            FastIDSet Excludeids = atask.getExcludeids();
            //find item part
            try {
                int size = dm.getNumUsers() - Excludeids.size() - 1;
//                long[] topItems = friendoffriendRecommend(testuser,size);
//                long[] topItems = followeroffriendRecommend(testuser,coveruser,size);
//                long[] topItems = PopRec(Excludeids,size);
                long[] topItems =similarityRecommend(dm,testuser,coveruser,size,Excludeids);
                for (int i = 0; i < size; i++) {
                    long topitem = topItems[i];
                    if (topitem == coveruser) {
                        atask.setRank(i + 1);
                        break;
                    }
                }
            } catch (TasteException e) {
                e.printStackTrace();
            }

            String output = atask.getTestuser() + "," + atask.getCoveruser() + "," + atask.getRank() + "," +
                            atask.getInteractions() + "," + atask.getTotalinteractions() + "\n";
            System.out.println(output);
            Excludeids.clear();
        }
    }
    public static void StatisticAndOutput(ArrayList<task> tasklist,Writer outputWriter ) throws IOException {
        HashMap<Long,Integer> UserRank=new HashMap<Long, Integer>();// for mrr
        double DCRP=0;
        int [] rankall=new int [tasklist.size()]; //for C@K
        int i=0;
        for(task atask:tasklist) {
            double DCRPi;
            long testuser=atask.getTestuser();
            int rank = atask.getRank();
            rankall[i++]=rank;
            String output_bias=atask.getTestuser()+","+atask.getCoveruser()+","+atask.getRank()+","+atask.getInteractions()+","+atask.getTotalinteractions()+"\n";
            double normalized_interaction=(double)atask.getInteractions()/atask.getTotalinteractions();
            double son= Math.pow(2,normalized_interaction)-1;
            double mother=Math.log(rank+1)/Math.log(2);
            DCRPi=son/mother;
            DCRP+=DCRPi;
            if(UserRank.containsKey(testuser)){
                if(UserRank.get(testuser)>rank){
                    UserRank.remove(testuser);
                    UserRank.put(testuser,rank);
                }
            }else{
                UserRank.put(testuser,rank);
            }
            outputWriter.write(output_bias);
            //  System.out.println("normal:"+output_bias);
        }
        DCRP/=tasklist.size();
        double MRR=0;
        for(long user:UserRank.keySet()){
            MRR+=(double)1/UserRank.get(user);
        }
        int [] coverK={5,10,20,50,100,200};
        int []N=new int [6];
        for(int rank:rankall){
            for(int j=0;j<6;j++){
                if(rank<=coverK[j]){
                    N[j]++;
                }
            }
        }
        for(int j=0;j<6;j++){
            double coverage=(double)N[j]/tasklist.size();
            System.out.println(coverage);
        }
        MRR/=UserRank.size();
        System.out.println(DCRP);
        System.out.println(MRR);
        System.out.println("User size:"+UserRank.size());
    }
    private static ArrayList<task> generate_task(HashSet<Long> SocialIDset,DataModel socialdm,DataModel dm,HashMap<Long,Long> postwriter) throws
                                                                                                                                          TasteException {
        ArrayList<task> tasklist=new ArrayList<task>();
        //generate task
        for (long testuser : SocialIDset) {
            //check if this user have to experiment
            HashMap<Long, Integer> interactions = new HashMap<Long, Integer>();
            int totalinteractions = 0;
            FastIDSet friendids = socialdm.getItemIDsFromUser(testuser);
            FastIDSet itemids = new FastIDSet();
            try {
                itemids = dm.getItemIDsFromUser(testuser);
            } catch (TasteException e) {
                //donothing
            }
            boolean effective = false;
            FastIDSet replyees = new FastIDSet();
            for (long item : itemids) {//  only experiment on those who reply exceed effectivelimit
                float score = 0;
                try {
                    score = dm.getPreferenceValue(testuser, item);
                } catch (TasteException e) {
                    e.printStackTrace();
                }
                if (score > 0) {//reply
                    long writer = postwriter.get(item);
                    replyees.add(writer);
                    //construct interaction number map
                    if (interactions.containsKey(writer)) {
                        interactions.put(writer, interactions.get(writer) + 1);
                        totalinteractions++;
                    } else {
                        interactions.put(writer, 1);
                        totalinteractions++;
                    }
                }
            }
            if (replyees.size() > replyee_effectivelimit&&friendids.size()>friend_effectivelimit) {
                effective = true;
            }
            if (effective) {
                for (long coveruser : interactions.keySet()) {  //change replyees or friendids
                    if (coveruser != testuser) {
                        int interaction = interactions.get(coveruser);
                        if (interactions.containsKey(coveruser)) {
                            FastIDSet exclude = replyees.clone();
                            exclude.remove(coveruser);
                            exclude.add(testuser);
                            task newtask = new task(testuser, coveruser, exclude, interaction, totalinteractions);
                            tasklist.add(newtask);
                        }
                    }
                }
            }
        }
        return tasklist;
    }
}
