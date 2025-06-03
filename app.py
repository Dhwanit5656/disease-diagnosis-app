from ast import dump
from json import load
from fastapi import FastAPI, Request, HTTPException # type: ignore
from fastapi.responses import JSONResponse, HTMLResponse # type: ignore 
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
import re
from fuzzywuzzy import process # type: ignore
import uvicorn # type: ignore
from joblib import dump, load # type: ignore
import os
from typing import Dict, List

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load and prepare data
data = pd.read_csv("data.csv")
data['Symptoms'] = data['Symptoms'].apply(lambda x: [s.strip().lower() for s in re.split(',|;', x)])
mlb = MultiLabelBinarizer()
symptoms_encoded = pd.DataFrame(mlb.fit_transform(data['Symptoms']), columns=mlb.classes_)
data_encoded = pd.concat([data['Disease'], symptoms_encoded], axis=1)
symptoms_list = symptoms_encoded.columns

# Load or train model
MODEL_PATH = "model.joblib"
if os.path.exists(MODEL_PATH):
    model = load(MODEL_PATH)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data_encoded[symptoms_list], data_encoded['Disease'])
    dump(model, MODEL_PATH)

# Disease information with improved normalization
DISEASE_INFO = {
    "Influenza": {
        "description": "A contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs.",
        "severity": "Moderate to Severe",
        "treatment": ["Rest", "Hydration", "Antiviral medications (oseltamivir, zanamivir)", 
                     "Pain relievers (ibuprofen, acetaminophen)"],
        "when_to_seek_help": "Difficulty breathing, persistent fever (>3 days), severe muscle pain, or dehydration",
        "prevention": ["Annual flu vaccine", "Frequent hand washing", "Avoiding close contact with sick individuals"],
        "complications": ["Pneumonia", "Bronchitis", "Sinus infections", "Worsening of chronic conditions"]
    },
    "Common Cold": {
        "description": "Viral infection of the upper respiratory tract affecting the nose and throat.",
        "severity": "Mild",
        "treatment": ["Rest", "Hydration", "Saline nasal drops", "Over-the-counter cold medicines",
                     "Throat lozenges", "Warm liquids"],
        "when_to_seek_help": "Symptoms lasting more than 10 days, high fever (>101.3°F), or difficulty breathing",
        "prevention": ["Hand hygiene", "Avoid touching face", "Disinfect surfaces"],
        "complications": ["Ear infection", "Sinusitis", "Asthma flare-ups"]
    },
    "COVID-19": {
        "description": "Respiratory illness caused by the SARS-CoV-2 virus, ranging from mild to severe.",
        "severity": "Mild to Critical",
        "treatment": ["Rest", "Hydration", "Fever reducers", "Antiviral medications (Paxlovid, Remdesivir)",
                     "Monoclonal antibodies (for high-risk patients)"],
        "when_to_seek_help": "Trouble breathing, persistent chest pain, confusion, inability to stay awake, bluish lips/face",
        "prevention": ["Vaccination", "Mask-wearing in crowded places", "Good ventilation", "Isolation when sick"],
        "complications": ["Pneumonia", "ARDS", "Blood clots", "Long COVID", "Multisystem inflammatory syndrome"]
    },
    "Pneumonia": {
        "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid.",
        "severity": "Moderate to Severe",
        "treatment": ["Antibiotics (bacterial)", "Antiviral (viral)", "Fever reducers", "Cough medicine",
                     "Hospitalization for severe cases"],
        "when_to_seek_help": "Difficulty breathing, fever >102°F, persistent cough with pus",
        "prevention": ["Pneumococcal vaccine", "Flu vaccine", "Good hygiene", "Not smoking"],
        "complications": ["Bacteremia", "Lung abscess", "Pleural effusion", "Respiratory failure"]
    },
     "Asthma": {
        "description": "It is a chronic condition where the airways become inflamed and narrowed, causing difficulty in breathing. It often leads to wheezing, coughing, and tightness in the chest",
        "severity": "Moderate to Severe",
        "treatment": ["Inhalers (bronchodilators", "steroids), nebulizers"," avoiding triggers (like allergens, pollution)"],
        "when_to_seek_help": " If wheezing, shortness of breath, or chest tightness becomes severe or if symptoms don't improve with medication",
    },
      "Bronchitis": {
        "description": "It is inflammation of the bronchial tubes, leading to a cough that produces mucus. It can be acute or chronic and is often caused by viral infections.",
        "severity": "Moderate to Severe",
        "treatment": ["Rest, fluids"," expectorants", "antibiotics (if bacterial)"," inhalers for breathing"],
        "when_to_seek_help": " If symptoms persist for more than 3 weeks or if breathing becomes difficult",
    },
      " Sinusitis ": {
        "description": "It  is an inflammation of the sinus cavities, often due to a viral infection or allergies, causing facial pain, nasal congestion, and headaches",
        "severity": "Moderate to Severe",
        "treatment": [" Nasal decongestants", "saline nasal spray", "antibiotics (for bacterial infections)", "pain relievers"],
        "when_to_seek_help": "  If pain is severe, symptoms last more than 10 days, or if there is a high fever",
    },
      "Migraine": {
        "description": "A migraine is a severe headache often accompanied by nausea, vomiting, and sensitivity to light. It may be triggered by factors like stress, certain foods, or hormonal changes",
        "severity": "Moderate to Severe",
        "treatment": ["Pain relievers, anti-nausea medications, lifestyle changes, avoiding triggers"],
        "when_to_seek_help": "  If migraines become more frequent or severe, or if there are unusual neurological symptoms like vision changes.",
    },
      "Hypertension": {
        "description": "Hypertension, or high blood pressure, is a condition where the force of the blood against the artery walls is consistently too high, potentially causing damage to the heart and other organs.",
        "severity": "Moderate to Severe",
        "treatment": [" Blood pressure-lowering medications (e.g., ACE inhibitors, beta-blockers)", "lifestyle changes like diet and exercise."],
        "when_to_seek_help": "If blood pressure remains high despite treatment, or if symptoms like headaches, dizziness, or shortness of breath occur",
    },
      "Diabetes": {
        "description": "Diabetes is a metabolic disease that causes high blood sugar levels. It can be Type 1 (insulin-dependent) or Type 2 (lifestyle-related), leading to various health complications if not controlled",
        "severity": "Moderate to Severe",
        "treatment": ["Insulin injections (Type 1)", "oral medications (Type 2)", "a controlled diet, and regular blood sugar monitoring"],
        "when_to_seek_help": "  If blood sugar levels remain out of control, or if symptoms like increased thirst, fatigue, or frequent urination persist.",
    },
      "Hypothyroidism": {
        "description": "Hypothyroidism occurs when the thyroid gland doesn’t produce enough thyroid hormones, leading to a slowed metabolism.",
        "severity": "Moderate to Severe",
        "treatment": ["Thyroid hormone replacement therapy, regular monitoring of thyroid function"],
        "when_to_seek_help": "If symptoms like fatigue, weight gain, or depression worsen or don’t improve with treatment.",
    },
      "Hyperthyroidism": {
        "description": "Hyperthyroidism is when the thyroid gland produces too much thyroid hormone, causing an accelerated metabolism",
        "severity": "Moderate to Severe",
        "treatment": ["Antithyroid medications, radioactive iodine therapy, or surgery to remove part of the thyroid"],
        "when_to_seek_help": "  If symptoms like rapid heartbeat, unexplained weight loss, or anxiety become severe.",
    },
      "Anemia": {
        "description": "Anemia is a condition where you don't have enough healthy red blood cells to carry oxygen throughout your body, leading to fatigue and other symptoms.",
        "severity": "Moderate to Severe",
        "treatment": [" Iron supplements, vitamin B12 injections, changes in diet (iron-rich foods)."],
        "when_to_seek_help": " If symptoms such as extreme fatigue, pale skin, or shortness of breath persist.",
    },
      "Arthritis": {
        "description": "Arthritis is inflammation of the joints, causing pain, stiffness, and swelling. It can be due to various types like osteoarthritis or rheumatoid arthritis.",
        "severity": "Moderate to Severe",
        "treatment": ["Pain relievers (NSAIDs), joint support, physical therapy, sometimes surgery."],
        "when_to_seek_help": "  If joint pain becomes unbearable, or if there is visible swelling or deformity.",
    },
      "Osteoporosis": {
        "description": "Osteoporosis is a condition where bones become weak and brittle, making them more prone to fractures.",
        "severity": "Moderate to Severe",
        "treatment": ["Calcium and vitamin D supplements, bisphosphonates, weight-bearing exercises."],
        "when_to_seek_help": " If there is a fracture after a minor fall or injury, or if you experience back pain or loss of height.",
    },
      "Gastroenteritis": {
        "description": "Gastroenteritis is the inflammation of the stomach and intestines, typically caused by viral or bacterial infections, leading to diarrhea, vomiting, and abdominal cramps.",
        "severity": "Moderate to Severe",
        "treatment": [" Hydration (oral rehydration solutions), anti-nausea medications, antibiotics (if bacterial)."],
        "when_to_seek_help": " If there is severe dehydration, blood in stool, or prolonged symptoms lasting more than 48 hours.",
    },
      "Peptic Ulcer": {
        "description": "A peptic ulcer is an open sore that develops on the lining of the stomach or the upper part of the small intestine, often due to an infection with H. pylori or long-term use of nonsteroidal anti-inflammatory drugs (NSAIDs).",
        "severity": "Moderate to Severe",
        "treatment": [" Proton pump inhibitors (PPIs), antibiotics (for H. pylori), avoiding irritants (alcohol, spicy food)."],
        "when_to_seek_help": "If there is severe stomach pain, nausea, or vomiting blood.",
    },
      "Hepatitis": {
        "description": "Hepatitis refers to the inflammation of the liver, often caused by viral infections, alcohol consumption, or certain medications.",
        "severity": "Moderate to Severe",
        "treatment": [" Antiviral medications, lifestyle changes, liver function monitoring."],
        "when_to_seek_help": " If jaundice, abdominal pain, or persistent fatigue occurs.",
    },
      "Cirrhosis": {
        "description": "Cirrhosis is the scarring of the liver tissue caused by long-term damage, often due to chronic alcohol use or hepatitis infection.",
        "severity": "Moderate to Severe",
        "treatment": [" Treatment of underlying cause (e.g., antiviral treatment for hepatitis), managing complications, liver transplant."],
        "when_to_seek_help": " If you experience severe fatigue, yellowing of the skin (jaundice), or swelling in the legs",
    },
      "Kidney Stones": {
        "description": "Kidney stones are hard deposits made of minerals and salts that form inside the kidneys, causing severe pain, especially when passing through the urinary tract.",
        "severity": "Moderate to Severe",
        "treatment": ["  Pain relief, increased fluid intake, medications to help pass the stone, and sometimes surgery."],
        "when_to_seek_help": " If experiencing severe pain, blood in urine, or difficulty urinating.",
    },
      "Urinary Tract Infection (UTI)": {
        "description": "A UTI is an infection in any part of the urinary system (kidneys, bladder, urethra), commonly caused by bacteria entering the urinary tract.",
        "severity": "Moderate to Severe",
        "treatment": ["  Antibiotics", "increased fluid intake", "pain relievers."],
        "when_to_seek_help": " If symptoms like painful urination, frequent urge to urinate, or blood in the urine occur.",
    },
      "Prostatitis": {
        "description": "Prostatitis is the inflammation of the prostate gland, which can cause pelvic pain, urinary issues, and sexual dysfunction.",
        "severity": "Moderate to Severe",
        "treatment": [" Antibiotics (if bacterial)", "pain relievers", "warm baths."],
        "when_to_seek_help": "If experiencing pain in the pelvic area, difficulty urinating, or fever.",
    },
    "Liver Disease": {
        "description": "Liver disease refers to various conditions affecting the liver, including fatty liver, hepatitis, cirrhosis, and liver cancer.",
        "severity": "Moderate to Severe",
        "treatment": [" Treatment depends on the underlying cause (e.g., antiviral drugs, alcohol cessation)."],
        "when_to_seek_help": "If experiencing yellowing of the skin (jaundice), severe fatigue, or abdominal swelling.",
    },
    "Pancreatitis": {
        "description": "Pancreatitis is inflammation of the pancreas, usually caused by gallstones or heavy alcohol use, leading to severe abdominal pain.",
        "severity": "Moderate to Severe",
        "treatment": ["  Fasting to allow the pancreas to rest", "pain relief, IV fluids",  "sometimes surgery."],
        "when_to_seek_help": " If experiencing sudden, severe abdominal pain, vomiting, or fever.",
    },
    "Gallstones": {
        "description": " Gallstones are hardened deposits of bile that can form in the gallbladder, causing pain, nausea, or blockage in the bile ducts.",
        "severity": "Moderate to Severe",
        "treatment": [" Surgery to remove the gallbladder (cholecystectomy)", "pain management."],
        "when_to_seek_help": " If experiencing sudden sharp pain in the upper abdomen or yellowing of the skin (jaundice).",
    },
    "Appendicitis": {
        "description": "Appendicitis is an inflammation of the appendix, often resulting in severe pain in the lower right abdomen, and can lead to rupture if untreated.",
        "severity": "Severe",
        "treatment": ["Surgical removal of the appendix (appendectomy)", "antibiotics"],
        "when_to_seek_help": "If experiencing sharp pain in the lower right abdomen, nausea, or fever."
    },
    "Diverticulitis": {
        "description": "Diverticulitis occurs when small, bulging pouches in the digestive tract become inflamed or infected, leading to abdominal pain and fever.",
        "severity": "Moderate to Severe",
        "treatment": ["Antibiotics", "dietary changes", "surgery in severe cases"],
        "when_to_seek_help": "If you have persistent abdominal pain, fever, or rectal bleeding."
    },
    "Crohn's Disease": {
        "description": "Crohn's Disease is a type of inflammatory bowel disease (IBD) causing chronic inflammation of the gastrointestinal tract, often resulting in diarrhea and abdominal pain.",
        "severity": "Chronic / Moderate to Severe",
        "treatment": ["Anti-inflammatory medications", "immunosuppressants", "dietary management", "surgery if needed"],
        "when_to_seek_help": "If experiencing ongoing diarrhea, abdominal pain, or weight loss."
    },
    "Ulcerative Colitis": {
        "description": "Ulcerative colitis is a chronic inflammatory condition affecting the colon and rectum, leading to bloody diarrhea and abdominal cramping.",
        "severity": "Moderate to Severe",
        "treatment": ["Anti-inflammatory drugs", "immune system suppressors", "surgery in severe cases"],
        "when_to_seek_help": "If you notice blood in your stool, weight loss, or persistent fatigue."
    },
    "Celiac Disease": {
        "description": "Celiac disease is an autoimmune disorder triggered by consuming gluten, causing damage to the small intestine lining and nutrient absorption issues.",
        "severity": "Mild to Moderate (Chronic)",
        "treatment": ["Strict lifelong gluten-free diet"],
        "when_to_seek_help": "If symptoms persist despite a gluten-free diet or if signs of malnutrition occur."
    },
    "Food Poisoning": {
        "description": "Food poisoning is caused by consuming contaminated food or water, leading to sudden gastrointestinal symptoms like nausea, vomiting, and diarrhea.",
        "severity": "Mild to Moderate",
        "treatment": ["Hydration", "rest", "antibiotics in bacterial cases"],
        "when_to_seek_help": "If symptoms last more than 2 days, or if there is high fever or blood in stools."
    },
    "Irritable Bowel Syndrome": {
        "description": "IBS is a common digestive disorder that affects the large intestine, causing abdominal discomfort, bloating, and altered bowel habits.",
        "severity": "Mild to Moderate (Chronic)",
        "treatment": ["Dietary adjustments", "stress management", "medications to control symptoms"],
        "when_to_seek_help": "If abdominal pain is severe or accompanied by weight loss or rectal bleeding."
    },
    "Hemorrhoids": {
        "description": "Hemorrhoids are swollen veins in the rectum or anus that cause discomfort, itching, and rectal bleeding.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical creams", "warm baths", "minimally invasive procedures for severe cases"],
        "when_to_seek_help": "If experiencing significant bleeding or persistent pain."
    },
    "Colon Cancer": {
        "description": "Colon cancer begins in the large intestine and may present with changes in bowel habits, blood in stool, and unexplained weight loss.",
        "severity": "Severe",
        "treatment": ["Surgery", "chemotherapy", "radiation therapy"],
        "when_to_seek_help": "If you have persistent changes in bowel habits, rectal bleeding, or weight loss."
    },
    "Lung Cancer": {
        "description": "Lung cancer originates in the lungs and often presents with persistent cough, chest pain, and weight loss. It is commonly linked to smoking.",
        "severity": "Severe",
        "treatment": ["Surgery", "chemotherapy", "radiation therapy", "targeted therapy"],
        "when_to_seek_help": "If you experience a persistent cough, coughing up blood, or unexplained weight loss."
    },
    "Breast Cancer": {
        "description": "Breast cancer is the uncontrolled growth of cells in breast tissue, often presenting as a lump, changes in breast shape, or nipple discharge.",
        "severity": "Severe",
        "treatment": ["Surgery", "radiation therapy", "chemotherapy", "hormonal therapy"],
        "when_to_seek_help": "If you find a lump in the breast, notice changes in breast shape or nipple discharge."
    },
    "Prostate Cancer": {
        "description": "Prostate cancer affects the prostate gland in men, often causing urinary symptoms or pelvic pain in advanced stages.",
        "severity": "Moderate to Severe",
        "treatment": ["Surgery", "radiation therapy", "hormonal therapy"],
        "when_to_seek_help": "If experiencing frequent or difficult urination or pelvic discomfort."
    },
    "Skin Cancer": {
        "description": "Skin cancer involves the abnormal growth of skin cells, often due to UV exposure. It may appear as new or changing moles or skin lesions.",
        "severity": "Varies (from Mild to Severe)",
        "treatment": ["Surgical removal", "cryotherapy", "topical medications", "radiation therapy"],
        "when_to_seek_help": "If you notice new skin growths, changes in moles, or non-healing sores."
        },
    "Leukemia": {
        "description": "Leukemia is a cancer of the blood-forming tissues, leading to abnormal white blood cell production and symptoms like frequent infections and bruising.",
        "severity": "Severe",
        "treatment": ["Chemotherapy", "radiation therapy", "stem cell transplant"],
        "when_to_seek_help": "If you experience frequent infections, easy bruising, or persistent fatigue."
    },
    "Lymphoma": {
        "description": "Lymphoma is a cancer of the lymphatic system, often causing swollen lymph nodes, night sweats, and fatigue.",
        "severity": "Moderate to Severe",
        "treatment": ["Chemotherapy", "radiation therapy", "immunotherapy"],
        "when_to_seek_help": "If you have persistent swollen lymph nodes, fever, or night sweats."
    },
    "Melanoma": {
        "when_to_seek_help": "If you notice dark, irregular, or changing moles on your skin."
    },
    "Multiple Sclerosis": {
        "description": "Multiple sclerosis is a chronic autoimmune disorder that affects the central nervous system, causing muscle weakness and vision issues.",
        "severity": "Moderate to Severe (Chronic)",
        "treatment": ["Immunosuppressive medications", "physical therapy", "symptom management"],
        "when_to_seek_help": "If you experience sudden vision problems, muscle weakness, or numbness."
    },
    "Alzheimer's Disease": {
        "description": "Alzheimer's disease is a progressive neurological disorder that leads to memory loss, confusion, and difficulty with thinking and speaking.",
        "severity": "Severe (Progressive)",
        "treatment": ["Medications to slow progression", "cognitive therapy", "supportive care"],
        "when_to_seek_help": "If memory issues interfere with daily life or there is confusion about time or place."
    },
    "Parkinson's Disease": {
        "description": "Parkinson's disease is a progressive neurological disorder that affects movement, causing tremors, stiffness, and difficulty with balance and coordination.",
        "severity": "Moderate to Severe (Chronic)",
        "treatment": ["Medications like levodopa", "physical therapy", "deep brain stimulation in advanced cases"],
        "when_to_seek_help": "If experiencing tremors, slowed movements, or balance issues interfering with daily activities."
    },
    "Epilepsy": {
        "description": "Epilepsy is a neurological disorder marked by recurring seizures due to abnormal brain activity, often resulting in convulsions and temporary loss of awareness.",
        "severity": "Moderate to Severe (Varies)",
        "treatment": ["Anti-seizure medications", "lifestyle adjustments", "surgery in some cases"],
        "when_to_seek_help": "If experiencing seizures or loss of consciousness for the first time."
    },
    "Stroke": {
        "description": "Stroke occurs when blood flow to a part of the brain is interrupted, leading to brain damage. Symptoms include numbness, confusion, and difficulty speaking or walking.",
        "severity": "Severe (Emergency)",
        "treatment": ["Clot-dissolving medication", "surgery", "rehabilitation therapy"],
        "when_to_seek_help": "Immediately seek emergency help if sudden numbness, confusion, or trouble speaking occurs."
    },
    "Meningitis": {
        "description": "Meningitis is the inflammation of the membranes surrounding the brain and spinal cord, typically due to infection. It can cause headache, fever, and stiff neck.",
        "severity": "Severe (Can be life-threatening)",
        "treatment": ["Antibiotics or antivirals", "hospitalization", "supportive care"],
        "when_to_seek_help": "If experiencing high fever, stiff neck, and sudden headache, especially with nausea or confusion."
    },
    "Encephalitis": {
        "description": "Encephalitis is an inflammation of the brain, often caused by viral infections, and may lead to confusion, fever, seizures, or loss of consciousness.",
        "severity": "Severe (Potentially life-threatening)",
        "treatment": ["Antiviral medications", "supportive care", "hospitalization"],
        "when_to_seek_help": "Seek help immediately if symptoms like confusion, seizures, or sudden changes in behavior occur."
    },
    "Polio": {
        "description": "Polio is a viral disease that can cause paralysis, fever, and muscle weakness. It primarily affects children and is preventable by vaccination.",
        "severity": "Severe",
        "treatment": ["Supportive care", "physical therapy", "ventilator support in severe cases"],
        "when_to_seek_help": "If muscle weakness, especially in the legs, or breathing difficulty occurs."
    },
    "Dengue": {
        "description": "Dengue is a mosquito-borne viral infection causing high fever, severe headaches, joint and muscle pain, and rash.",
        "severity": "Moderate to Severe",
        "treatment": ["Pain relievers (acetaminophen)", "hydration", "hospital care for severe cases"],
        "when_to_seek_help": "If experiencing high fever with bleeding, severe abdominal pain, or persistent vomiting."
    },
    "Malaria": {
        "description": "Malaria is a mosquito-borne infectious disease characterized by fever, chills, and sweating, caused by parasites transmitted through mosquito bites.",
        "severity": "Moderate to Severe",
        "treatment": ["Antimalarial medications", "supportive care"],
        "when_to_seek_help": "If fever, chills, and flu-like symptoms appear after travel to malaria-prone areas."
    },
    "Typhoid": {
        "description": "Typhoid is a bacterial infection caused by Salmonella typhi, resulting in prolonged fever, weakness, and sometimes a rash.",
        "severity": "Moderate",
        "treatment": ["Antibiotics", "hydration", "rest"],
        "when_to_seek_help": "If high fever persists for several days, especially after consuming unsafe food or water."
    },
    "Cholera": {
        "description": "Cholera is a bacterial infection caused by ingesting contaminated water or food, leading to severe watery diarrhea and dehydration.",
        "severity": "Severe (Potentially life-threatening)",
        "treatment": ["Oral rehydration salts (ORS)", "intravenous fluids", "antibiotics in severe cases"],
        "when_to_seek_help": "Seek immediate medical attention if experiencing profuse watery diarrhea and signs of dehydration."
    },
    "Leptospirosis": {
        "description": "Leptospirosis is a bacterial infection spread through contact with water contaminated by animal urine, causing fever, muscle pain, and red eyes.",
        "severity": "Moderate to Severe",
        "treatment": ["Antibiotics (doxycycline or penicillin)", "supportive care", "hospitalization in severe cases"],
        "when_to_seek_help": "If experiencing sudden fever, muscle pain, and conjunctivitis after exposure to contaminated water."
    },
    "Zika Virus": {
        "description": "Zika virus is a mosquito-borne infection causing mild symptoms like rash, fever, and joint pain, but it can cause birth defects if contracted during pregnancy.",
        "severity": "Mild (but serious in pregnancy)",
        "treatment": ["Rest", "hydration", "pain relievers such as acetaminophen"],
        "when_to_seek_help": "If pregnant and exposed to Zika, or if experiencing rash and fever after travel to an affected area."
    },
    "Ebola": {
        "description": "Ebola is a rare but deadly viral infection that causes severe bleeding, organ failure, and can lead to death if untreated.",
        "severity": "Critical",
        "treatment": ["Supportive care (fluids and electrolytes)", "oxygen therapy", "experimental antiviral treatments"],
        "when_to_seek_help": "Seek emergency medical help if experiencing high fever, bleeding, and recent travel to or contact with an Ebola-affected area."
    },
    "HIV/AIDS": {
        "description": "HIV is a virus that weakens the immune system by attacking white blood cells. If untreated, it can progress to AIDS, making the body vulnerable to infections and cancers.",
        "severity": "Chronic (Potentially life-threatening)",
        "treatment": ["Antiretroviral therapy (ART)", "regular monitoring and supportive treatments for infections"],
        "when_to_seek_help": "Seek help if at risk of exposure or experiencing unexplained weight loss, fatigue, or frequent infections."
    },
    "Syphilis": {
        "description": "Syphilis is a sexually transmitted bacterial infection that progresses in stages, causing sores, rashes, and, if untreated, severe damage to organs.",
        "severity": "Moderate to Severe (if untreated)",
        "treatment": ["Antibiotics (usually penicillin)", "regular follow-up testing"],
        "when_to_seek_help": "Seek medical evaluation if experiencing painless sores or rashes, especially after unprotected sex."
    },
    "Gonorrhea": {
        "description": "Gonorrhea is a common sexually transmitted infection caused by bacteria, affecting the genitals, rectum, and throat, and can lead to infertility if untreated.",
        "severity": "Moderate",
        "treatment": ["Antibiotics", "partner treatment to prevent reinfection"],
        "when_to_seek_help": "Seek help if experiencing painful urination, unusual discharge, or if exposed to a partner diagnosed with an STI."
    },
    "Chlamydia": {
        "description": "Chlamydia is a bacterial sexually transmitted infection that may cause genital ulcers and swollen lymph nodes, and can lead to reproductive complications if untreated.",
        "severity": "Mild to Moderate",
        "treatment": ["Antibiotics", "screening and partner treatment"],
        "when_to_seek_help": "Seek help if experiencing genital discomfort, unusual discharge, or after unprotected sex."
    },
    "Herpes": {
        "description": "Herpes is a viral infection causing painful blisters or sores around the mouth or genitals. It is chronic and can recur periodically.",
        "severity": "Chronic (Mild to Moderate)",
        "treatment": ["Antiviral medications (e.g., acyclovir)", "pain management"],
        "when_to_seek_help": "Seek help if experiencing recurrent blisters or sores, especially after unprotected sexual activity."
    },
    "HPV Infection": {
        "description": "Human papillomavirus (HPV) is a common viral infection transmitted through skin-to-skin contact that can cause warts and is linked to cervical and other cancers.",
        "severity": "Varies (Usually Mild, but can be Severe)",
        "treatment": ["Topical treatments for warts", "monitoring for precancerous changes", "HPV vaccination"],
        "when_to_seek_help": "Seek help if warts appear or if advised after an abnormal Pap test."
    },
    "Measles": {
        "description": "Measles is a highly contagious viral disease characterized by fever, cough, rash, and can lead to serious complications like pneumonia or brain inflammation.",
        "severity": "Moderate to Severe",
        "treatment": ["Supportive care (rest, fluids)", "vitamin A supplements", "fever reducers"],
        "when_to_seek_help": "Seek help immediately if a red rash follows high fever, especially in unvaccinated individuals."
    },
    "Mumps": {
        "description": "Mumps is a viral infection that causes painful swelling of the salivary glands, along with fever and headache, and may lead to complications like hearing loss.",
        "severity": "Moderate",
        "treatment": ["Supportive care (rest, pain relievers)", "ice packs on swollen glands"],
        "when_to_seek_help": "Seek help if swelling of the jaw or cheeks occurs, especially with fever."
    },
    "Rubella": {
        "description": "Rubella, also known as German measles, is a viral infection that causes rash, joint pain, and mild fever, and is especially dangerous in pregnancy.",
        "severity": "Mild (but Severe in pregnancy)",
        "treatment": ["Supportive care (rest, fluids, fever control)"],
        "when_to_seek_help": "Seek help if a rash develops along with joint pain or if exposed during pregnancy."
    },
    "Chickenpox": {
        "description": "Chickenpox is a contagious viral infection causing an itchy, blister-like rash along with fever, and can lead to complications in adults or those with weak immune systems.",
        "severity": "Mild to Moderate",
        "treatment": ["Antiviral medications (in severe cases)", "calamine lotion and antihistamines for itch relief", "rest and fluids"],
        "when_to_seek_help": "Seek help if blisters become infected, if fever is high, or for adults or infants showing symptoms."
    },
    "Shingles": {
        "description": "Shingles is a viral infection caused by the reactivation of the chickenpox virus, leading to a painful rash and burning sensation, often on one side of the body.",
        "severity": "Moderate to Severe",
        "treatment": ["Antiviral medications", "pain relievers", "topical creams"],
        "when_to_seek_help": "Seek help if you experience a painful rash, especially if near the eyes or if you're immunocompromised."
    },
    "Tetanus": {
        "description": "Tetanus is a serious bacterial infection that affects the nervous system, causing jaw stiffness, muscle spasms, and difficulty breathing.",
        "severity": "Severe",
        "treatment": ["Tetanus immune globulin (TIG)", "antibiotics", "muscle relaxants", "supportive care"],
        "when_to_seek_help": "Seek immediate help after a deep or dirty wound if not vaccinated or if showing symptoms of muscle rigidity."
    },
    "Rabies": {
        "description": "Rabies is a deadly viral infection transmitted through the saliva of infected animals, affecting the brain and spinal cord, and is almost always fatal once symptoms appear.",
        "severity": "Critical",
        "treatment": ["Immediate post-exposure prophylaxis (PEP)", "rabies immunoglobulin", "vaccination"],
        "when_to_seek_help": "Seek emergency help immediately after any potential exposure to a rabid animal, even if no symptoms are present."
    },
    "Anthrax": {
        "description": "Anthrax is a rare but serious bacterial infection that can occur through the skin, inhalation, or ingestion, causing sores, breathing difficulties, or intestinal issues.",
        "severity": "Severe to Critical",
        "treatment": ["Antibiotics (ciprofloxacin, doxycycline)", "antitoxins", "supportive care"],
        "when_to_seek_help": "Seek help if exposed to infected animals or suspicious powders and symptoms like skin sores or respiratory issues develop."
    },
    "Lyme Disease": {
        "description": "Lyme disease is a tick-borne illness caused by bacteria, often marked by a bullseye rash, joint pain, and flu-like symptoms, and can cause chronic complications if untreated.",
        "severity": "Mild to Moderate (can become Severe if untreated)",
        "treatment": ["Antibiotics (doxycycline, amoxicillin)", "anti-inflammatory medications"],
        "when_to_seek_help": "Seek help if bitten by a tick and symptoms like rash or joint pain appear within a few weeks."
    },
    "Tuberculosis": {
        "description": "Tuberculosis (TB) is a contagious bacterial infection primarily affecting the lungs, but it can spread to other parts of the body, causing chronic cough, night sweats, and weight loss.",
        "severity": "Moderate to Severe",
        "treatment": ["Antibiotics (usually for 6-9 months)", "directly observed therapy (DOT)"],
        "when_to_seek_help": "Seek help if you have a persistent cough, night sweats, unexplained weight loss, or contact with someone diagnosed with TB."
    },
    "Plague": {
        "description": "Plague is a serious bacterial infection transmitted through fleas from rodents, causing swollen lymph nodes, fever, and, in severe cases, septicemia or pneumonia.",
        "severity": "Severe to Critical",
        "treatment": ["Antibiotics (streptomycin, gentamicin)", "supportive care"],
        "when_to_seek_help": "Seek immediate medical attention if you experience sudden fever, swollen lymph nodes, or have been in contact with rodents or fleas."
    },
    "Leprosy": {
        "description": "Leprosy, or Hansen's disease, is a chronic infectious disease that causes skin lesions, nerve damage, and muscle weakness, spread by droplets from the nose or mouth.",
        "severity": "Mild to Moderate",
        "treatment": ["Antibiotics (multi-drug therapy)"],
        "when_to_seek_help": "Seek help if you have persistent skin lesions, numbness, or muscle weakness, particularly if in close contact with infected individuals."
    },
    "Hantavirus": {
        "description": "Hantavirus is a viral disease spread by rodents, leading to fever, muscle aches, and difficulty breathing, sometimes causing hantavirus pulmonary syndrome (HPS).",
        "severity": "Severe",
        "treatment": ["Supportive care", "ventilatory support"],
        "when_to_seek_help": "Seek immediate medical help if you experience flu-like symptoms and have had exposure to rodents or their droppings."
    },
    "Yellow Fever": {
        "description": "Yellow fever is a viral infection transmitted by mosquitoes, leading to fever, jaundice, and organ failure in severe cases.",
        "severity": "Moderate to Severe",
        "treatment": ["Supportive care", "fluid therapy"],
        "when_to_seek_help": "Seek immediate help if you develop fever and jaundice after being bitten by mosquitoes in areas with yellow fever transmission."
    },
    "Rocky Mountain Spotted Fever": {
        "description": "Rocky Mountain spotted fever is a tick-borne disease caused by bacteria, leading to fever, rash, and muscle aches, often appearing in the spring or summer.",
        "severity": "Severe",
        "treatment": ["Antibiotics (doxycycline)"],
        "when_to_seek_help": "Seek medical attention if you have a tick bite and develop fever, rash, or muscle aches."
    },
    "Tularemia": {
        "description": "Tularemia is a bacterial infection spread through infected animals, ticks, or insect bites, leading to fever, swollen lymph nodes, and ulcers.",
        "severity": "Moderate",
        "treatment": ["Antibiotics (streptomycin, gentamicin)"],
        "when_to_seek_help": "Seek help if you have been exposed to rabbits, rodents, or insect bites and develop fever or skin ulcers."
    },
    "Brucellosis": {
        "description": "Brucellosis is a bacterial infection transmitted from animals to humans, often causing fever, sweating, muscle pain, and fatigue.",
        "severity": "Moderate",
        "treatment": ["Antibiotics (doxycycline, rifampin)"],
        "when_to_seek_help": "Seek medical attention if you experience fever, fatigue, and have had contact with livestock or animal products."
    },
    "Psoriasis": {
        "description": "Psoriasis is a chronic autoimmune condition that speeds up skin cell turnover, leading to red, scaly patches of skin, often on the scalp, elbows, and knees.",
        "severity": "Mild to Severe",
        "treatment": ["Topical treatments (steroids, vitamin D analogs)", "phototherapy", "biologics"],
        "when_to_seek_help": "Seek help if you experience persistent, red, scaly patches on the skin, especially if they are painful or affecting daily life."
    },
    "Eczema": {
        "description": "Eczema is an inflammatory skin condition characterized by itchy, red, and inflamed skin. It can be triggered by allergens, irritants, or stress.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical corticosteroids", "moisturizers", "avoidance of triggers"],
        "when_to_seek_help": "Seek help if your eczema symptoms worsen, become infected, or if home treatments don't improve your condition."
    },
    "Vitiligo": {
        "description": "Vitiligo is a skin condition where the immune system attacks pigment-producing cells, leading to white patches of skin, often on the hands, face, or areas exposed to the sun.",
        "severity": "Mild",
        "treatment": ["Topical corticosteroids", "light therapy", "cosmetic cover-ups"],
        "when_to_seek_help": "Seek medical help if you notice white patches of skin that expand over time or cause emotional distress."
    },
    "Rosacea": {
        "description": "Rosacea is a chronic skin condition causing redness, visible blood vessels, and sometimes acne-like breakouts, typically on the face.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical medications (metronidazole, azelaic acid)", "oral antibiotics", "laser therapy"],
        "when_to_seek_help": "Seek help if you notice persistent redness, acne-like pimples, or irritation on your face."
    },
    "Acne": {
        "description": "Acne is a common skin condition that causes pimples, blackheads, and cysts, usually on the face, back, and shoulders. It occurs when hair follicles are clogged with oil and dead skin cells.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical treatments (benzoyl peroxide, salicylic acid)", "oral antibiotics", "retinoids"],
        "when_to_seek_help": "Seek help if over-the-counter treatments aren't effective, or if acne causes scarring."
    },
    "Alopecia": {
        "description": "Alopecia is an autoimmune disorder where the body's immune system attacks hair follicles, leading to hair loss, usually on the scalp.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical corticosteroids", "minoxidil", "immunotherapy"],
        "when_to_seek_help": "Seek help if you experience sudden or patchy hair loss or notice bald patches on your scalp or body."
    },
    "Dandruff": {
        "description": "Dandruff is a common scalp condition where dead skin flakes off the scalp, often accompanied by itching and dryness.",
        "severity": "Mild",
        "treatment": ["Anti-dandruff shampoos (zinc pyrithione, selenium sulfide)", "moisturizing scalp treatments"],
        "when_to_seek_help": "Seek help if dandruff is persistent or worsens, causing scalp irritation or hair thinning."
    },
    "Fungal Infections": {
        "description": "Fungal infections occur when fungi infect the skin, nails, or mucous membranes, leading to symptoms like red, itchy patches or thickened nails.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical antifungals (clotrimazole, terbinafine)", "oral antifungals for severe cases"],
        "when_to_seek_help": "Seek medical help if you experience persistent itching, redness, or infections that don't respond to home treatment."
    },
    "Ringworm": {
        "description": "Ringworm is a fungal infection characterized by a ring-shaped rash that is red, itchy, and scaly, often occurring on the skin, scalp, or feet.",
        "severity": "Mild to Moderate",
        "treatment": ["Topical antifungals (clotrimazole, miconazole)", "oral antifungals for severe cases"],
        "when_to_seek_help": "Seek help if the rash spreads, becomes severe, or doesn't improve with over-the-counter treatments."
    },
    "Scabies": {
        "description": "Scabies is a contagious skin infestation caused by mites, leading to intense itching, rashes, and blisters, often between the fingers, wrists, and genital area.",
        "severity": "Moderate to Severe",
        "treatment": ["Topical scabicides (permethrin cream)", "oral ivermectin for severe cases"],
        "when_to_seek_help": "Seek medical attention if you experience intense itching, rashes, or if symptoms spread despite treatment."
    },
    "Lupus": {
        "description": "Lupus is an autoimmune disease where the body's immune system attacks its own tissues, causing inflammation and damage to the skin, joints, kidneys, and other organs.",
        "severity": "Moderate to Severe",
        "treatment": ["Immunosuppressants (hydroxychloroquine, methotrexate)", "NSAIDs for inflammation", "steroids for flare-ups"],
        "when_to_seek_help": "Seek medical help if you experience a butterfly-shaped rash on the face, joint pain, or unexplained fatigue."
    },
    "Scleroderma": {
        "description": "Scleroderma is an autoimmune condition that causes thickening and hardening of the skin and connective tissues, sometimes affecting organs like the heart or lungs.",
        "severity": "Moderate to Severe",
        "treatment": ["Immunosuppressants", "physical therapy", "medications to control symptoms"],
        "when_to_seek_help": "Seek help if you notice skin hardening, joint pain, or difficulty breathing, as it may affect internal organs."
    },
    "Rheumatic Fever": {
        "description": "Rheumatic fever is an inflammatory disease that can develop after a streptococcal throat infection, causing fever, joint pain, and potential heart damage.",
        "severity": "Moderate to Severe",
        "treatment": ["Antibiotics (penicillin)", "NSAIDs for joint pain", "steroids for inflammation"],
        "when_to_seek_help": "Seek help if you have had a sore throat followed by fever, joint pain, or chest pain."
    },
    "Gout": {
        "description": "Gout is a form of arthritis caused by excess uric acid buildup, leading to sudden, severe pain and swelling, typically in the joints of the feet, especially the big toe.",
        "severity": "Moderate",
        "treatment": ["NSAIDs for pain", "uric acid-lowering medications (allopurinol)", "dietary changes to reduce uric acid levels"],
        "when_to_seek_help": "Seek help if you experience sudden, severe joint pain, especially in the big toe, or recurring flare-ups."
    },
    "Fibromyalgia": {
        "description": "Fibromyalgia is a chronic condition characterized by widespread musculoskeletal pain, fatigue, and sleep disturbances, often triggered by stress or illness.",
        "severity": "Moderate",
        "treatment": ["Pain management (NSAIDs, antidepressants)", "physical therapy", "stress management techniques"],
        "when_to_seek_help": "Seek help if you experience persistent pain, fatigue, or sleep disturbances that interfere with daily life."
    },
    "Depression": {
        "description": "Depression is a mood disorder causing persistent sadness, loss of interest in activities, and feelings of hopelessness, impacting daily functioning.",
        "severity": "Moderate to Severe",
        "treatment": ["Antidepressant medications", "therapy (CBT, talk therapy)", "lifestyle changes (exercise, social support)"],
        "when_to_seek_help": "Seek help if you experience persistent sadness, lack of motivation, or thoughts of self-harm or suicide."
    },
    "Anxiety Disorders": {
        "description": "Anxiety disorders involve excessive worry, nervousness, or fear that can interfere with daily activities, leading to symptoms like racing heart, sweating, and restlessness.",
        "severity": "Mild to Severe",
        "treatment": ["Cognitive Behavioral Therapy (CBT)", "anti-anxiety medications (benzodiazepines, SSRIs)", "relaxation techniques"],
        "when_to_seek_help": "Seek help if anxiety interferes with your daily life or leads to panic attacks, excessive worry, or physical symptoms."
    },
    "Bipolar Disorder": {
        "description": "Bipolar disorder is a mood disorder characterized by extreme mood swings, including periods of mania (high energy, impulsivity) and depression.",
        "severity": "Severe",
        "treatment": ["Mood stabilizers (lithium)", "antidepressants", "psychotherapy"],
        "when_to_seek_help": "Seek medical help if you experience severe mood swings, impulsive behavior, or episodes of extreme depression or mania."
    },
    "Schizophrenia": {
        "description": "Schizophrenia is a severe mental disorder that affects thinking, emotions, and behavior, often leading to hallucinations, delusions, and disorganized thoughts.",
        "severity": "Severe",
        "treatment": ["Antipsychotic medications", "therapy (cognitive behavioral therapy)", "supportive care"],
        "when_to_seek_help": "Seek help if you experience hallucinations, delusions, or disorganized thinking that disrupt daily life."
    },
    "Obsessive-Compulsive Disorder": {
        "description": "Obsessive-Compulsive Disorder (OCD) involves repetitive thoughts (obsessions) and behaviors (compulsions) that are difficult to control and interfere with daily life.",
        "severity": "Moderate",
        "treatment": ["Cognitive Behavioral Therapy (CBT)", "medications (SSRIs)"],
        "when_to_seek_help": "Seek help if obsessive thoughts and compulsive behaviors interfere with daily functioning or cause distress."
    },
    "Autism Spectrum Disorder": {
        "description": "Autism Spectrum Disorder (ASD) is a developmental disorder affecting communication, behavior, and social interaction, with symptoms varying widely from mild to severe.",
        "severity": "Mild to Severe",
        "treatment": ["Behavioral therapy", "speech therapy", "educational support"],
        "when_to_seek_help": "Seek help if a child exhibits difficulties in communication, social interactions, or displays repetitive behaviors."
    },
    "ADHD": {
        "description": "Attention-Deficit/Hyperactivity Disorder (ADHD) is a neurodevelopmental disorder characterized by inattention, hyperactivity, and impulsivity, affecting children and adults.",
        "severity": "Mild to Moderate",
        "treatment": ["Stimulant medications (methylphenidate)", "behavioral therapy", "organizational strategies"],
        "when_to_seek_help": "Seek help if inattention, impulsivity, or hyperactivity interfere with school, work, or relationships."
    },
    "PTSD": {
        "description": "Post-Traumatic Stress Disorder (PTSD) occurs after exposure to traumatic events, leading to flashbacks, nightmares, and heightened anxiety.",
        "severity": "Severe",
        "treatment": ["Cognitive Behavioral Therapy (CBT)", "EMDR therapy", "medications (SSRIs, SNRIs)"],
        "when_to_seek_help": "Seek help if you experience flashbacks, nightmares, or intense anxiety following trauma."
    },
    "Insomnia": {
        "description": "Insomnia is a sleep disorder characterized by difficulty falling or staying asleep, leading to daytime fatigue and irritability.",
        "severity": "Mild to Moderate",
        "treatment": ["Cognitive Behavioral Therapy for Insomnia (CBT-I)", "sleep hygiene", "medications (benzodiazepines, melatonin)"],
        "when_to_seek_help": "Seek help if insomnia persists for several weeks or affects daily functioning."
    },
    "Narcolepsy": {
        "description": "Narcolepsy is a chronic sleep disorder characterized by excessive daytime sleepiness, sudden sleep attacks, and, in some cases, cataplexy (sudden muscle weakness).",
        "severity": "Moderate to Severe",
        "treatment": ["Stimulant medications", "antidepressants for cataplexy", "lifestyle modifications (scheduled naps, regular sleep routines)"],
        "when_to_seek_help": "Seek help if you experience sudden, uncontrollable sleep episodes or daytime fatigue that interferes with daily life."
    }
}


def normalize_disease_name(name: str) -> str:
    """Normalize disease names for consistent lookup"""
    name = name.strip().title()
    variations = {
        "Covid19": "COVID-19",
        "Covid-19": "COVID-19",
    "description": "Melanoma is a serious type of skin cancer that develops in pigment-producing cells and often appears as dark or irregular moles.",
        "severity": "Severe",
        "treatment": ["Surgical excision", "immunotherapy", "chemotherapy"],
            "Flu": "Influenza"
    }
    return variations.get(name, name)

def get_disease_info(disease_name: str) -> Dict:
    """Improved disease info lookup with fallbacks"""
    normalized_name = normalize_disease_name(disease_name)
    
    # Exact match
    if normalized_name in DISEASE_INFO:
        return DISEASE_INFO[normalized_name]
    
    # Fuzzy match
    best_match, score = process.extractOne(normalized_name, DISEASE_INFO.keys())
    if score > 80:
        return DISEASE_INFO[best_match]
    
    # Fallback for unknown diseases
    return {
        "description": f"No detailed information available for {disease_name}",
        "severity": "Unknown",
        "treatment": ["Consult a healthcare provider"],
        "when_to_seek_help": "If symptoms persist or worsen"
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/symptoms/list")
async def list_symptoms():
    return {"symptoms": sorted(symptoms_list.tolist())}

@app.get("/symptoms/suggest")
async def suggest_symptoms(query: str, limit: int = 5):
    matches = process.extract(query, symptoms_list, limit=limit)
    return {"suggestions": [match[0] for match in matches if match[1] > 60]}

@app.get("/symptoms/common_combos")
async def common_combinations():
    top_combos = data['Symptoms'].value_counts().head(5).index.tolist()
    return {"combinations": [", ".join(combo) for combo in top_combos]}

@app.get("/disease/{name}")
async def disease_info(name: str):
            return get_disease_info(name)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "").lower()
    symptoms = [s.strip() for s in re.split(',|;', user_input) if s.strip()]
    
    if len(symptoms) < 3:
        return JSONResponse(
            content={"error": "Please enter at least 3 symptoms separated by commas."},
            status_code=400
        )
    
    symptom_input = pd.DataFrame(0, index=[0], columns=symptoms_list)
    corrected_symptoms = []
    for symptom in symptoms:
        best_match, score = process.extractOne(symptom, symptoms_list)
        if score >= 80:
            symptom_input[best_match] = 1
            corrected_symptoms.append(best_match)
        else:
            return JSONResponse(
                content={"error": f"Invalid symptom: '{symptom}'. Did you mean {best_match}?"}
            )
    
    prediction_probabilities = model.predict_proba(symptom_input)[0]
    disease_probabilities = list(zip(model.classes_, prediction_probabilities))
    disease_probabilities.sort(key=lambda x: x[1], reverse=True)
    top_diseases = disease_probabilities[:3]
    
    response = "Top 3 potential diagnoses:\n\n"
    predictions = []
    for i, (disease, prob) in enumerate(top_diseases):
        response += f"{i+1}. {disease} ({prob:.1%} probability)\n"
        predictions.append({"disease": disease, "probability": prob})
    
    return JSONResponse(content={
        "response": response,
        "predictions": predictions
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)