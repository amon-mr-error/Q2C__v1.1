DATASET = [
  {
    "query": "What is the main objective of the document? in one line",
    "ground_truth": "The main objective of the document is to discuss various aspects of machine learning, including its definition, types such as unsupervised and semi-supervised learning, and applications like predictive analysis and deep learning.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "Which deep learning architectures are used in the study? in one line",
    "ground_truth": "The study uses InceptionV3, InceptionResNetV2, and MobileNet architectures for skin disease classification.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "What are the main phases of the proposed system?in one line",
    "ground_truth": "The main phases of the proposed system are training, validating/testing, and potential future standardization for preliminary skin disease diagnosis (Source — test2.pdf, Page 3, Page 7).",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "Why is skin disease diagnosis considered difficult? in one line",
    "ground_truth": "Skin disease diagnosis is considered difficult due to the complexities of skin and the wide range of environmental and geographical factors that cause variations in these diseases.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "How does the system make final predictions? in one line",
    "ground_truth": "The system makes final predictions by using a validation/testing set to tune the parameters and assess the effectiveness and efficiency of the model (Source — test2.pdf, Page 3).",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "What dataset split is used for training and testing? in one line",
    "ground_truth": "The dataset is divided into 90% for training and 10% for validation/testing.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "What role does transfer learning play in the model? in one line",
    "ground_truth": "Transfer learning involves using a pre-trained model, such as Inception V3, which attains high accuracy in recognizing general materials by extracting features from input images and classifying them based on those features.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "What accuracy improvement does the proposed method achieve? in one line",
    "ground_truth": "The system achieves up to 88% accuracy using ensemble deep learning techniques.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "What machine learning techniques are used along with deep learning? in one line",
    "ground_truth": "Logistic regression and random forest are used for training and testing along with deep learning models.",
    "relevant_docs": ["test2.pdf"]
  },
  {
    "query": "What is the significance of ensemble learning in this study?",
    "ground_truth": "In this study, ensemble learning is significant for improving the efficiency of distinguishing different kinds of dermatological skin abnormalities. The method involves using various types of deep learning algorithms such as Inception_v3, MobileNet, Resnet, and Xception for feature extraction. After training these models, an ensemble is created by combining the outputs of these models using a voting-based approach. This ensemble learning technique further increases the efficiency of the model beyond what is achieved by individual models. Specifically, the study reports that using state-of-the-art architectures can increase efficiency up to 88 percent, and ensemble feature mapping can enhance this efficiency even more.",
    "relevant_docs": ["test2.pdf"]
  },

  {
    "query": "What is the purpose of the Appraisal Clause in the policy? in one line",
    "ground_truth": "The purpose of the Appraisal Clause in the policy is to allow either the insurer or the insured to demand a binding appraisal of damaged property in the event of a dispute regarding its value",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "How is a single outage defined under the Outage Clause? in one line",
    "ground_truth": "A single outage is defined as the period from the breakdown causing a shutdown until the unit is synchronized and achieves full load or operates for 72 hours since synchronization, whichever is earlier",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "What happens to the insurer’s obligations in case of bankruptcy of the insured? in one line",
    "ground_truth": "In the event of bankruptcy or insolvency of the insured, the insurer shall not be relieved of its obligations under the policy ",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "What does the Automatic Reinstatement of Sum Insured clause specify? in one line",
    "ground_truth": "The Automatic Reinstatement of Sum Insured clause specifies that the basic sum insured under material damage remains at risk and shall not be reduced following a loss, as long as the aggregate of the sums paid and/or payable does not exceed a specified percentage of the sum insured mentioned in the schedule",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "What types of structures are covered under the Outbuilding Clause? in one line",
    "ground_truth": "The Outbuilding Clause covers walls, gates and fences, small outbuildings, extensions, annexes, exterior staircase and steel or iron frameworks, car parks, internal roadways including temporary roads, pavements, landscaping or any other structure situated on the insured premises ",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "How does the No Control Clause protect the insured? in one line",
    "ground_truth": "The No Control Clause protects the insured by ensuring that the insurance coverage is not adversely affected by any acts or omissions of tenants that are unknown to or beyond the control of the insured, as long as the insured notifies the company upon becoming aware of such acts",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "What is the role of transfer learning or pre-trained models in this policy document? in one line",
    "ground_truth": "Transfer learning or pre-trained models are not mentioned in this policy document",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "What are the conditions for currency conversion under the policy? in one line",
    "ground_truth": "The conditions for currency conversion under the policy are:For premium payments, the rate at the inception of each policy year.For premium adjustment, the rate at the expiry of each policy year.For loss settlement, the rate at the date of final settlement of loss",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "What is covered under the Loss of Keys/Changing Locks clause? in one line",
    "ground_truth": "The Loss of Keys/Changing Locks clause covers the cost of replacing keys and locks or modifying the locking mechanism to any strongroom, safe, or money receptacle in the event of such keys or locks having been stolen",
    "relevant_docs": ["test1.pdf"]
  },
  {
    "query": "How does the Fraud and Forfeiture Clause affect claims? in one line",
    "ground_truth": "The Fraud and Forfeiture Clause voids the policy and exempts the insurer from payment if an insured party makes a fraudulent claim or false statement, but the insurer may cancel the policy with a 30-day notice and provide a pro-rata refund",
    "relevant_docs": ["test1.pdf"]
  }
]