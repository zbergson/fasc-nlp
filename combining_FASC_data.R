#This file combines coded data from S17, S18, and F17. 
#Participants with no username or empty text responses or duplicate entries are removed.

library(readr)
library(dplyr)

#all data files are in our Dropbox folder - YOU MAY NEED TO SET DIFFERENT PATH
setwd("~/Dropbox/FASC_NLP/S17_F17_S18_Data/files_for_combining/")

#load S17 Data
FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2 <- read_csv("SC_FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2.csv")

# FASC_S17_FINAL_DATA_SET_fixed_just <- read_csv("~/Dropbox/FASC_EF_Study_2018/S17_Data/FASC_S17_FINAL_DATA_SET_fixed_just.csv")
# FASC_EF_Games_Data_blank <- read_csv("~/Dropbox/FASC_EF_Study_2018/S17_Data/FASC_EF_Games Data_blank.csv")
# test1 <- anti_join(FASC_EF_Games_Data_blank, FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2, by = c("sipasID", "username", "condition_value"))
# test2 <- anti_join(FASC_S17_FINAL_DATA_SET_fixed_just,FASC_EF_Games_Data_blank, by = c("sipasID", "username", "condition_value"))
# test3 <- anti_join(FASC_S17_FINAL_DATA_SET_fixed_just, FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2, by = c("sipasID", "username", "condition_value", "lagtime_ms"))
# test4 <- anti_join(FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2, FASC_S17_FINAL_DATA_SET_fixed_just, by = c("sipasID", "username", "condition_value", "lagtime_ms"))

#load F17 and S18 data
F17_S18_FASCPILOT2_Coded_Spelling_Checked <- read_csv("F17_S18_FASCPILOT2_Coded_Spelling_Checked.csv")

#load data that we've alread used for mental terms logic testing and development
X100_entries_FASC_F17_S18_Alg_Human_Comparison <- read_csv("100_entries_FASC_F17_S18_Alg_Human_Comparison.csv")
X20_percent_reviewed_regression_2_fasc_nlp_analysis <- read_csv("20_percent_reviewed_regression_2_fasc_nlp_analysis.csv")

F17_S18_FASCPILOT2_Coded_Spelling_Checked$username <- as.factor(as.integer(F17_S18_FASCPILOT2_Coded_Spelling_Checked$username))
X100_entries_FASC_F17_S18_Alg_Human_Comparison$username <- as.factor(as.integer(X100_entries_FASC_F17_S18_Alg_Human_Comparison$username))
X20_percent_reviewed_regression_2_fasc_nlp_analysis$username <- as.factor(as.integer(X20_percent_reviewed_regression_2_fasc_nlp_analysis$username))
FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2$username <- as.factor(as.integer(FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2$username))

#find the 100 rows in the F17 and S18 that were used for first 100 test (Round 1) and assign them Round 1
match_100 <- semi_join(F17_S18_FASCPILOT2_Coded_Spelling_Checked,X100_entries_FASC_F17_S18_Alg_Human_Comparison, by = c("sipasID", "username", "condition_value", "response_iter","sesID"))
match_100$Round <- 1

#find the rows in the F17 and S18 that were used for first 20% test (Round 2) and assign them Round 2
match_20_percent <- inner_join(F17_S18_FASCPILOT2_Coded_Spelling_Checked,X20_percent_reviewed_regression_2_fasc_nlp_analysis, by = c("sipasID", "username", "condition_value", "response_iter","sesID"))
match_20_percent$Round <- 2

#transfer F19 scoring to df with spell checked F17 / S18 responses
match_20_percent <- select(match_20_percent, sipasID, username, condition_value, response_iter,
                           spelling_checked_response_text, ends_with(".x"), L_M_column_match, sesID, 
                             Mental_terms_tot.y, Ment_stat_just_tot.y, Com_first_resp.y, 
                             -Mental_terms_tot.x, -Ment_stat_just_tot.x, -Com_first_resp.x, 
                           Round, Human_Checked, Year, sesID)

match_20_percent <- match_20_percent %>% rename_all(~sub("\\.x", "", .x))
match_20_percent <- match_20_percent %>% rename_all(~sub("\\.y", "", .x))

#reorder columns to match order in F17_S18_FASCPILOT2_Coded_Spelling_Checked
match_20_percent<-select(match_20_percent, colnames(F17_S18_FASCPILOT2_Coded_Spelling_Checked))

#separate data that has not been assigned a reliability round
remaining <- anti_join(F17_S18_FASCPILOT2_Coded_Spelling_Checked,match_20_percent, by = c("sipasID", "username", "condition_value", "response_iter", "sesID"))
remaining2 <- anti_join(remaining,match_100, by = c("sipasID", "username", "condition_value", "response_iter", "sesID"))

#file now has rows with appropriate Round designation
F17_S18_FASCPILOT2_Coded_Spelling_Checked_with_rounds <- rbind(match_100, match_20_percent, remaining2)

#select subset of columns to match the S17 data
sub_F17_S18_FASCPILOT2_Coded_Spelling_Checked_with_rounds<- select(F17_S18_FASCPILOT2_Coded_Spelling_Checked_with_rounds, 
              sipasID, username, condition_value, response_iter, response_text, spelling_checked_response_text, Resp_tot, 
              Mental_terms_tot, Ment_stat_just_tot, Com_first_resp, lagtime_ms, response_datetime,
              Round, Year, Human_Checked)

#Create S17 Round and Human Checked columns, assign years to data
FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2$Round <- NA
FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2$Year <- "S17"
FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2$Human_Checked <- NA
sub_F17_S18_FASCPILOT2_Coded_Spelling_Checked_with_rounds$Year <- "F17_S18"

#Combine data for S17, F17, and S18
FASC_S17_F17_S18 <- rbind(sub_F17_S18_FASCPILOT2_Coded_Spelling_Checked_with_rounds,FASC_S17_FINAL_DATA_SET_fixed_just_NA_time_removed_2)

#Remove participants with no username or empty text responses or duplicate entries
FASC_S17_F17_S18 <- FASC_S17_F17_S18%>%
  filter(!is.na(username))%>%
  filter(!is.na(spelling_checked_response_text))%>%
  group_by(sipasID, username, condition_value, response_iter, spelling_checked_response_text)%>%
  add_tally()%>%
  filter(row_number()==1)%>%
  as.data.frame()

#Separate rows with Round filled in - will be < 99+228 b/c cut username / responses w/ NAs
random_assign1_2 <- FASC_S17_F17_S18%>%filter(!is.na(Round))

#Separate rows with no Round filled in
FASC_S17_F17_S18_no_rounds <- FASC_S17_F17_S18%>%filter(is.na(Round))

#Separate by condition
FASC_S17_F17_S18_A <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="A")
FASC_S17_F17_S18_B <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="B")
FASC_S17_F17_S18_C <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="C")
FASC_S17_F17_S18_D <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="D")
FASC_S17_F17_S18_E <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="E")
FASC_S17_F17_S18_F <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="F")
FASC_S17_F17_S18_G <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="G")
FASC_S17_F17_S18_H <- FASC_S17_F17_S18_no_rounds%>%filter(condition_value=="H")
  
#BE CAREFUL WITH RUNNING THIS NEXT PART - the randomly assigned reliability rounds will be different each time!

#Assign 20% of each condition to Round 3, Round 4, etc. Had to do with replace =  TRUE b/c otherwise it complains.
values <- c("3", "4", "5", "6", "7")
FASC_S17_F17_S18_A$Round <- sample(values, nrow(FASC_S17_F17_S18_A), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_B$Round <- sample(values, nrow(FASC_S17_F17_S18_B), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_C$Round <- sample(values, nrow(FASC_S17_F17_S18_C), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_D$Round <- sample(values, nrow(FASC_S17_F17_S18_D), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_E$Round <- sample(values, nrow(FASC_S17_F17_S18_E), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_F$Round <- sample(values, nrow(FASC_S17_F17_S18_F), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_G$Round <- sample(values, nrow(FASC_S17_F17_S18_G), TRUE, prob = c(.2,.2,.2,.2,.2))
FASC_S17_F17_S18_H$Round <- sample(values, nrow(FASC_S17_F17_S18_H), TRUE, prob = c(.2,.2,.2,.2,.2))

#reassemble S17, F17, and S18 with all rounds designated
FASC_S17_F17_S18_all_Rounds <- rbind(random_assign1_2, FASC_S17_F17_S18_A, 
                                     FASC_S17_F17_S18_B,
                                     FASC_S17_F17_S18_C,
                                     FASC_S17_F17_S18_D,
                                     FASC_S17_F17_S18_E,
                                     FASC_S17_F17_S18_F,
                                     FASC_S17_F17_S18_G,
                                     FASC_S17_F17_S18_H)

#count how many of each condition in each round
tallies <- FASC_S17_F17_S18_all_Rounds%>% #should be about 630 in each Round 3 - 7
  group_by(Round, condition_value)%>%
  tally()%>%
  as.data.frame()

#WARNING: BE CAREFUL WITH WRITING - the randomly assigned reliability rounds will be different each time!
#last generated on 12-6-19 at 5:55PM
#write_csv(FASC_S17_F17_S18_all_Rounds, "../FASC_S17_F17_S18_all_Rounds_12-6-19.csv")