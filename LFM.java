package com.predictionmarketing.itemrecommend;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.*;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Created by eagle on 2014/9/18.
 *  * reviewed from 2015/11/17
 */

//find replyer and friends cover rate
//friends in rank list position
// ranking coverage rate
// eliminate unuse records

public  class LFM {

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

    //parameter
    protected static final int iteration =10;
    protected static final int replyee_effectivelimit=5;
    protected static final int friend_effectivelimit=5;
    protected static final int features=600;
    protected static final int Num_of_thread=1;

    public static void main(String[] args) throws Exception {

        // input
        DataModel replydm = new FileDataModel(new File("data/replyscn300.csv"));
        DataModel writerdm = new FileDataModel(new File("data/writerscn300.csv"));
        DataModel socialdm = new FileDataModel(new File("data/socialscn300.csv"));

       /* int dm_reply_size=0;
        int dm_noreply_size=0;
        LongPrimitiveIterator dit = dm.getUserIDs() ;
        while (dit.hasNext()) {
            long userid =  dit.nextLong();
            for (long itemid : dm.getItemIDsFromUser(userid)){
                if (dm.getPreferenceValue(userid,itemid)>0) {
                    dm_reply_size++;
                }else{
                    dm_noreply_size++;
                }
            }
        }*/

        //create social data model idset because user may have no friend ,and it will cause exception
        HashSet<Long> Social_IDset=new HashSet<Long>();
        LongPrimitiveIterator social_it=socialdm.getUserIDs();
        while (social_it.hasNext()) {
            long userID = social_it.nextLong();
            Social_IDset.add(userID);
        }

        HashMap<Long,Long> post_writer_table=new HashMap<Long, Long>();
        //construct post_writer_table
        LongPrimitiveIterator writer_it = writerdm.getUserIDs() ;
        while (writer_it.hasNext()) {
            long userid =  writer_it.nextLong();
            for (long item : writerdm.getItemIDsFromUser(userid)){
                post_writer_table.put(item, userid);
            }
        }
        //input and process data end
        System.out.println("start trainging");



        ArrayList<task> tasklist=generate_task( Social_IDset, socialdm, replydm, post_writer_table);
        System.out.println("Task generated end, there are "+tasklist.size()+" tasks(experiments)");

        //test
        ExecutorService executors = Executors.newFixedThreadPool(Num_of_thread);
        for(task atask:tasklist) {
            executors.submit(new userOperate(replydm,socialdm, writerdm, features, iteration, atask, post_writer_table));
        }
        executors.shutdown();
        try {
            executors.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException ee) {
            ee.printStackTrace();
        }
        //experiment end

        //start output result
        Writer outputWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream("reply-bias.csv"), "utf-8"));
        outputWriter.write("testuser,coveruser,rank,interactions,totalinteractions" + "\n");

        StatisticAndOutput(tasklist,outputWriter);
        outputWriter.close();
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

    public static long[] findtopofPACISsim(Factorization mf,long userid,int size,FastIDSet excludeids,int startindex) throws TasteException {
        long[] result = new long[size];

        HashMap<Long,Double> itemvaluemap=new HashMap<Long, Double>();
        double[]uservector=mf.getUserFeatures(userid);

        //compute score and store in itemvaluemap
        for(long i=0;i<mf.numUsers();i++) {
            if(!excludeids.contains(i)) {
                double score = 0.0;
                double[] itemvector = mf.getItemFeatures(i);
                for (int j = startindex; j < mf.numFeatures(); j++) {
                    score += uservector[j] * itemvector[j];
                }
                assert score<1.1;
                itemvaluemap.put(i, score);

            }
        }


        SortedSet<Map.Entry<Long,Double>> sorted_map =entriesSortedByValues(itemvaluemap);
        if(sorted_map.size()!=size) {
            System.out.println(size);
            System.out.println(sorted_map.size());
        }
        int count=0;
        for (Map.Entry<Long,Double> entry  : sorted_map) {
            result[count++]=entry.getKey();
        }
        /*System.out.println("FIRST:"+sorted_map.first().getKey()+":"+sorted_map.first().getValue()+"; result:"+result[0]);
        System.out.println("LAST:"+sorted_map.last().getKey()+":"+sorted_map.last().getValue()+"; result:"+result[size-1]);*/
        return result;
    }
    static class userOperate implements Runnable {
        private  DataModel dm;
        private  DataModel socialdm;
        private  DataModel writerdm;
        private  int features;
        private  int iteration;
        private task atask;
        private  HashMap<Long,Long> postwriter;

        public userOperate(DataModel dm,DataModel socialdm, DataModel writerdm,int features,int iteration,task atask,HashMap<Long,Long> postwriter) {
            this.dm=dm;
            this.socialdm=socialdm;
            this.writerdm=writerdm;
            this.features=features;
            this.iteration=iteration;
            this.atask=atask;
            this.postwriter=postwriter;
        }

        public void run() {
            double task_start=System.currentTimeMillis();
            long testuser=atask.getTestuser();
            long coveruser=atask.getCoveruser();
            FastIDSet Excludeids=atask.getExcludeids();
            try {
//                SUSSGDFactorizer f  = new SUSSGDFactorizer(dm, socialdm, writerdm, features, iteration, testuser,
 //                                                          coveruser, postwriter);
//                Factorizer f  = new CliMF(dm, socialdm, writerdm, features, iteration, testuser,
//                                                           coveruser, postwriter);
                // run normal MF
                Factorizer f = new RatingSGDFactorizer(dm,socialdm, writerdm, features, iteration,  testuser,coveruser,postwriter);
//                Factorizer f = new SVDPPFactorizer(dm,socialdm, writerdm, features, iteration,  testuser,coveruser,postwriter);
                Factorization mf = f.factorize();
                //find item part
                int size = dm.getNumUsers() - Excludeids.size();
                long[] topItems_bias= new long[0];
                try {
                    topItems_bias = findtopofPACISsim(mf, testuser, size, Excludeids, 0);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                for (int i = 0; i < size; i++) {
                    long topitem = topItems_bias[i];
                    if (topitem == coveruser) {
                        atask.setRank(i+1);
                        break;
                    }
                }

            } catch (TasteException e) {
                System.out.println("error");
                e.printStackTrace();
            }
//            String output=atask.getTestuser()+","+atask.getCoveruser()+","+atask.getRank()+","+atask.getInteractions()+","+atask.getTotalinteractions()+"\n";
            // System.out.println(output);
            Excludeids.clear();
            double task_end=System.currentTimeMillis();
            task_end=(task_end-task_start)/60000;
                System.out.println(task_end+"mins");
        }
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