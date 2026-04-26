# The prevalence and correlates of depression and anxiety symptoms among older adults in Shenzhen, China

<https://doi.org/10.5061/dryad.bnzs7h4j1>

---

This study employed a cross-sectional population-based survey research design and used a multistage random whole-group sampling method to select representative samples from 10 districts in Shenzhen, China . From October 2020 to February 2021, a total of 5,372 participants were invited to participate in the study, and were asked to complete a mental health assessment questionnaire. Of these, 41 participants were excluded due to poor questionnaire completion quality, leaving 5,331 participants (99.2%) for data analysis. A structured questionnaire was utilized to gather information on the participants' sociodemographic characteristics and health assessment parameters. The questionnaire included sociodemographic characteristics, lifestyle, physical health, and mental health.
Investigators utilized various means such as posters, radio broadcasts, and media channels, to promote mental health surveys in the sampled communities and to gain the understanding, attention, and cooperation of the community. Investigators then made individual appointments according to the sample list, explained the purpose of the survey, the process, and the benefits of participation to the respondents, and obtained consent. A specific time for the face-to-face household survey was then determined. As mental health-related issues are sensitive information, we asked all participants to complete the survey in a private one-on-one setting.

**Datasets included:**

**1. sociodemographic characteristics**

*   Education, represent the different levels of education of the participants, categorical variable, levels 1-5, 1= Primary school and below, 2= Junior high school, 3= Highschool/vocational secondary school, 4= College, 5= Master and above.

*   Marriage, represent different marital statuses of participants, categorical variable, levels = 1-2, 1=Unmarried/divorced/widowed, 2=Married.

*   Suffering from chronic diseases, is based on the diagnosis of chronic disease by second level and above hospitals, categorical variable, levels = 1-2, 1=NO, 2=YES.

*   Monthly personal income, refers to a participant's average monthly income, categorical variable, levels 1-5, 1= ≤$216.74, 2=$216.89-$433.63, 3=$433.78-$722.82, 4=$722.96-$1156.59, 5=≥$1156.74.

*   Drinking, refers to a participant's drinking behaviour, categorical variable, levels 1-3, 1=Nondrinker, 2=Ex-drinker (previous drinkers who currently abstain from alcohol), 3=Current drinker (1 or more drinks per week).

*   Smoking, refers to a participant's smoking behaviour, categorical variable, levels 1-3, 1=Nonsmoker, 2=Ex-smoker (individuals with a previous history of smoking who currently abstain), 3=Current smoker.

*   Health status in the past year, categorical variable, levels 1-5, 1=Good, 2=Relatively good, 3=Ordinary, 4=Relatively poor, 5=Poor.

*   Sleep duration, continuous variables, refers to the participant's average sleep duration at night in the most recent year.

**2.mental health**

*   Mean score of the PHQ-9, continuous variables. The Patient Health Questionnaire Depression Scale-9 item (PHQ-9) was utilized to evaluate the occurrence of depressive symptoms in the participants. The PHQ-9 comprises nine items that measure the respondent's depressive state and severity in the past year, with each item rated on a four-point scale from 0 (not at all) to 3 (almost every day). The total score ranges from 0-27, with higher scores indicating more severe depression.
*   Depressive symptoms, we used a PHQ-9 score of 5 as the cut-off point, with a score greater than or equal to 5 indicating the presence of depressive symptoms and a score less than 5 indicating the absence of depressive symptoms. Categorical variable, levels 0-1, 0=NO, 1=YES.
*   Mean score of the GAD-7, continuous variables. The Generalized Anxiety Disorder 7-item scale (GAD-7) was utilized to assess the occurrence of anxiety symptoms in the participants. (25) Respondents recall their anxiety status and severity assessment within the past year, rating each item on a four-point scale from 0 (not at all) to 3 (almost every day), with a total score range of 0-21. Higher scores indicate more severe anxiety in participants.
*   Anxiety symptoms, a GAD-7 score of 5 was used as the threshold, with scores greater than or equal to 5 indicating the presence of anxiety symptoms and scores less than 5 indicating the absence of anxiety symptoms. Categorical variable, levels 0-1, 0=NO, 1=YES.
*   Mean score of the AD8, continuous variables. The 8-item Ascertain Dementia Questionnaire (AD8) was utilized to assess early mild cognitive impairment (MCI) on an eight-item scale. These items include diminished assertiveness, reduced engagement in hobbies, repetition of the same thing, difficulty in learning new things, forgetting the current year, difficulty handling complex financial matters, difficulty recalling appointments with others, and problems with memory and thinking. The total score ranges from 0 to 8, with higher scores indicating more severe cognitive impairment.
*   Mild cognitive impairment, a AD8 score of 2 was used as the nodal point, with scores greater than or equal to 2 indicating possible mild cognitive impairment and scores less than 2 indicating normal cognitive functioning. Categorical variable, levels 0-1, 0=NO, 1=YES.
*   Mean score of the CSID, continuous variables. The Brief Community Screening Instrument for Dementia (CSI-D) was utilized to assess the presence of early dementia among the participants. The scale includes seven cognitive items, which are ranked in descending order of difficulty as follows: describing the purpose of a hammer, naming the elbow, pointing to the window and then to the door, identifying the location of a nearby shop, identifying the current season, identifying the current week, and recalling three words after a delay. The total score on the scale ranges from 0 to 9, with higher scores indicating better cognitive functioning.
*   Early dementia, a CSID score of 7 was used as the threshold for early dementia, with scores greater than 7 indicating no evidence of early dementia and scores less than or equal to 7 indicating the presence of early dementia. Categorical variable, levels 0-1, 0=NO, 1=YES.
*   Insomnia (ISI), continuous variables. The Insomnia Severity Index (ISI) was utilized to evaluate the occurrence and severity of insomnia in the participants, consisting of seven items. (27) Respondents are asked to recall their insomnia symptoms in the past month. Each item is rated on a four-point scale ranging from 0 (not at all) to 3 (almost every day). The total score ranges from 0-21, with higher scores indicating more severe insomnia symptoms.
*   Insomnia, a score of 7 was chosen as the threshold, with scores greater than or equal to 7 indicating the presence of insomnia symptoms and scores less than 7 indicating the absence of insomnia symptoms. Categorical variable, levels 0-1, 0=NO, 1=YES.
*   Loneliness (ULS-6), continuous variables. A simplified version of the UCLA Loneliness Scale (ULS-6) was utilized to evaluate the discrepancy between the respondents' desire for social interaction and their actual level of interaction. The ULS-6 was translated and revised in Chinese and comprises six items, each rated on a four-point scale from 1 (never) to 4 (always). The total score ranges from 6 to 24, with higher scores indicating more severe loneliness.

Details for each dataset are provided in the CODE file.
 

\***Code/Software**

The statistical analysis was conducted using R version 4.1.0. The R package "compareGroups" was utilized for descriptive analysis. One-way linear regression was used to identify the factors associated with depressive symptoms and anxiety symptoms. The variables that were statistically significant in the univariate analysis were included in a multifactorial stepwise linear regression model to evaluate the relationship between depressive symptoms and anxiety symptoms. The analyses were performed using the R packages "car" and "MASS".
