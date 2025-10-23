---
title: "고대 최성준 교수님 VLA"
date: 2025-10-11
tags: [Robot, VLA, Physical AI]
toc: true
---
# Prelude

> “The world is its own best model.”
  Rodney Brooks (1990)

이 글은 최근 로보틱스 연구의 중심 화두로 떠오른 Physical AI에 대한 개인적인 생각과 질문들을 담고 있습니다. 먼저 분명히 하고 싶은 것은, 제가 던지는 질문들에 정답은 없다는 점입니다. 다만 서로의 이해 수준을 조금 더 맞추기 위해, Robotic Foundation Model (RFM), 구체적으로는 Vision-Language-Action (VLA) 모델, 의 발전 흐름을 살펴보고, 궁극적으로 인간형 로봇이 지향해야 할 물리적 지능의 본질을 함께 고찰하고자 합니다.

인간의 Physical Intelligence는 인지적 사고보다 훨씬 오래된 신체 기반의 지능으로, 불확실한 물리적 상호작용을 다루는 능력입니다. 이후 이 개념은 Physical AI라는 이름으로 확장되었으며, 구글의 SayCan, RT-1, RT-2로 이어진 연구를 통해 VLA 학습의 전환점이 마련되었습니다. 이어서 Physical Intelligence와 Skild AI 같은 새로운 기업의 등장, OpenVLA와 HPT 같은 학계의 연구, 그리고 NVIDIA의 GR00T N1과 Google DeepMind의 Gemini Robotics 등 빅테크 기업들의 참여를 통해, 로봇이 언어·시각·행동을 통합적으로 학습하는 방향으로 빠르게 발전하고 있습니다.

이 과정에서 언어 모델은 Physical AI의 핵심 구성 요소 중 하나로 자리 잡았습니다. 그러나 저는 언어나 추론 중심의 접근이 곧 Physical AI의 본질이라고 보지 않습니다. 언어는 제어의 대역폭이 낮으며, 정교한 조작(dexterous manipulation)에서는 오히려 시각과 촉각에 기반한 연속적(continuous)이고 강인한(robust) 제어가 핵심이라고 생각합니다. 결국 인간형 로봇의 지능은 불확실한 환경 속에서 인간과 유사한 손과 팔을 통해 구현되는 물리적 지능(Physical Intelligence)이 필수적이며, 앞으로는 이러한 방향의 연구에 더 많은 관심과 노력이 집중되어야 한다고 믿습니다.

# Physical Intelligence ?
> Moravec's paradox
"It is comparatively easy to make computers exhibit adult-level performance on intelligence tests or playing checkers, and difficult or impossible to give them the skills of a one-year-old when it comes to perception and mobility.” 
/ Hans Moravec (1988)

요즘 모두가 Physical AI를 이야기하고 있습니다. 그 말을 언제 처음 들었을까를 생각해보면, 제가 아는 한에서는 MIT의 김상배 교수님께서 발표에서 자주 사용하셨던 Physical Intelligence라는 표현이 떠오릅니다. 2022년, 그러니까 테슬라가 막 옵티머스 프로토타입을 공개했을 무렵, 김상배 교수님은 TED[1]에서 “Robots with Physical Intelligence”라는 제목으로 강연을 하셨습니다. 발표의 서두에서 교수님은 이렇게 말씀하십니다. “Physical Intelligence가 무엇일까요? 사실 저도 그것이 정확히 무엇인지는 모르겠습니다. 하지만 이 발표가 끝난 뒤에는 여러분이 제가 말하고자 하는 바를 이해할 수 있으면 좋겠습니다.” 이 영상을 처음 보았을 때 약간의 의문점이 들었던 것이 사실입니다. 생체모방 로보틱스를 연구하시고, 당시 로봇 치타(Cheetah)라는 세계에서 가장 빠른 사족보행 로봇을 만든 분이셨기에, 우리들이 일반적으로 생각하는 ‘지능(Intelligence)’이라는 단어와는 다소 거리가 있다고 생각했기 때문입니다.

이 발표에서 김상배 교수님께선 Physical Intelligence를 인간의 Cognitive Intelligence와 대비되는 개념으로 등장시킵니다. 우리가 더 익숙한 Cognitive Intelligence는 글쓰기, 수학, 체스와 같은 활동을 의미합니다. 이는 걷거나 손을 쓰는 신체 지능(Physical Intelligence)에 비해 인류 진화 과정에서 비교적 최근에 발달한 능력입니다. 또한 이러한 인지적 활동은 의식적 사고나 추론을 수반한다고 여겨져 상대적으로 ‘더 어렵다’고 평가받습니다. 그러나 이런 작업은 실행 과정에서 불확실성이 거의 없습니다. 예를 들어 체스 게임에서는 많은 생각을 하지만, 말을 옮길 때 물리적으로 실패할 가능성은 고려하지 않습니다. 

반면 Physical Intelligence는 전혀 다른 특성을 가집니다. 인간은 수백만 년에 걸쳐 생존을 위해 신체 지능을 진화시켜 왔습니다. 그래서 너무 자연스럽고 무의식적이어서 종종 과소평가되곤 합니다. 하지만 이 지능은 제어되지 않은 물리적 환경 속에서 매 순간 불확실성을 다룹니다. 예를 들어 주머니 속에서 물건을 꺼내는 단순한 행동을 생각해보면, 우리는 별생각 없이 손을 넣고 물건을 꺼내지만, 그 과정에는 마찰, 장애물, 손가락의 미세한 조정 같은 수많은 물리적 상호작용이 존재합니다. 흥미로운 점은 우리가 이러한 복잡한 물리 법칙을 수식으로 몰라도, 본능적으로 매우 높은 성공률로 이 동작을 수행한다는 것입니다.

이러한 작업들의 가장 큰 특징은 외부 환경과의 접촉이 매우 많다는 점입니다. 현재 공장에서는 작업의 자동화를 위해 수많은 로봇팔이 사용되고 있습니다. 한국의 경우 인구 1만 명당 1,012대의 로봇이 존재해, 세계에서 가장 높은 로봇 밀도를 가진 제조 강국으로 알려져 있습니다[2]. 그러나 이러한 로봇들은 물체를 파지(grasp)하는 경우를 제외하면 외부와의 접촉을 최소화하도록 설계되어 있습니다. 이들은 작업 속도와 반복 정밀도를 핵심 성능 지표로 삼기 때문에 일반적으로 높은 감속비의 액추에이터를 사용하며, 이러한 구조적 특성 때문에 한 번 충돌이 발생하면 큰 사고로 이어질 수밖에 없습니다. 이를 완화하기 위한 시도로 협업로봇(Collaborative Robot) 시장이 등장했지만, 여전히 기존 산업용 로봇과 유사한 폼팩터를 유지하며, 기본 목표 또한 충돌을 최소화하면서 안전하게 작업하는 것에 머무르고 있습니다. 반면 Physical Intelligence는 사람이 손으로 물건을 집고 조작하는 능력을 모방하려는 시도이기에, 고자유도(high-DOF) 로봇 손을 사용하고, 외부 환경과의 다중 접촉(multi-contact) 상황을 전제로 합니다.

사실 이 글에서 말하는 Physical Intelligence는 뒤에서 다룰 Physical AI와 초점이 조금 다르다고 생각합니다. 제 의견으로는 Physical AI가 물리 세계 전반에 적용 가능한 포괄적 인공지능을 가리킨다면, 김상배 교수님이 언급하신 Physical Intelligence는 사람이 무의식적으로 수행하는 반사적, 반응적(reactive) 행동에 더 가깝습니다. 그럼에도 서두에서 Physical Intelligence를 먼저 꺼낸 이유는, 현재의 Physical AI 논의가 이 지점, 신체 기반의 즉각적 상호작용과 정밀함 그리고 강건함, 을 종종 놓치고 있다고 느끼기 때문입니다. 그럼 이제 본격적으로 이야기를 이어가 보겠습니다.

# Robotic Foundation Model ?

> “Language is the source of misunderstanding.” 
-Antoine de Saint-Exupéry, The Little Prince (1943)

먼저 용어 정리를 하고 넘어가도록 하죠. 제가 좋아하는 말 중 하나는 어린 왕자의 저자 생텍쥐페리(Antoine de Saint-Exupéry)가 했던 “Language is the source of misunderstanding”입니다. 서로 다른 배경에서 자라왔기 때문에, 올바른 용어의 통일 없이는 유의미한 대화를 이어가기 어렵기 때문입니다. 제 이야기를 조금 해보고자 합니다. 저는 기억이 닿는 한 평생 로봇 연구를 하고 싶었습니다. ‘로봇이 아니면 안 된다’라는 거창한 꿈보다는, 할 수 있다면 로봇을 하자, 그리고 가능하다면 사람 곁에서 공존하는 로봇을 만들자라는 생각이 제 모토가 되어왔습니다. 학부 시절에는 병역특례로 청소로봇을 만드는 회사에서 일했고, 자연스럽게 대학원에서도 로보틱스를 연구하는 연구실로 진학했습니다. 박사 학위 주제 역시 로봇 모방 학습(Imitation Learning)이었습니다. 사실 그 시절에는 딥러닝을 포함한 기계학습 방법을 로봇 제어에 적용한다는 것 자체가 생소했습니다. 그래서 “왜 최적제어나 Model Predictive Control (MPC)을 쓰지 않았느냐”는 비판적인 리뷰를 종종 받기도 했습니다. 그만큼 ‘로봇’과 ‘학습’은 당시에는 잘 어울리지 않는 조합이었죠. 박사 학위를 받은 지 아직 10년도 채 되지 않았는데, 이제 세상이 학습 기반의 로보틱스로 빠르게 전환되는 것을 보고 있자면 개인적으로 참 신기할 따름입니다.

아마도 비교적 최근에 Physical AI라는 단어를 접하신 분들은 Robotic Foundation Model (RFM)이라는 표현도 함께 들어보셨을 것 같습니다. 이 단어는 로보틱스에만 국한된 개념이 아니라, 인공지능 분야에서 사용되는 Foundation Model[3]이라는 용어에 Robot을 붙인 형태라고 볼 수 있습니다. (제가 인공지능 전문가가 아니기 때문에, 일부 내용은 다를 수도 있다는 점 양해 부탁드립니다.) Foundation Model은 특정 문제를 해결하기 위해 만들어진 특화된 모델이 아니라, 다양한 문제에 범용적으로 활용될 수 있는 모델을 의미합니다. 예를 들어 OpenAI의 GPT 모델들이 대표적인 사례일 것입니다. 저 역시 지금 이 글을 작성하거나 평소에 코딩을 할 때 ChatGPT를 자주 활용하고 있습니다. Robotic Foundation Model 역시 특정 작업, 예를 들어 물체를 집어 옮기는 단순한 pick-and-place 작업에만 한정되지 않고, 로봇이 가위를 들어 종이를 자르는 것과 같은 보다 복잡하고 다양한 작업을 수행할 수 있기를 기대하는 개념입니다. 물론 2025년 10월 현재, 이러한 수준의 진정한 의미의 Robotic Foundation Model은 아직 존재하지 않는다고 생각합니다. 뒤에서 더 자세히 다루겠지만, 현재로서는 Physical Intelligence사의 π0.5[4] 같은 모델들이 그 초기 단계에 해당한다고 볼 수 있습니다.

# Vision-Language-Action (VLA) Model ?

Robotic Foundation Model은 아직 명확한 정의를 갖고 있지 않기 때문에, 저를 비롯해 이 분야를 연구하는 사람들은 대신 Vision-Language-Action(VLA) Model이라는 용어를 자주 사용합니다. 저 개인적으로는 이 단어 역시 컴퓨터 비전이나 자연어 처리에서 사용되던 Vision-Language Model에 로봇의 행동을 의미하는 Action이라는 단어가 추가된 형태라고 생각합니다. 따라서 사용되는 방법론 역시 유사합니다. 이미 학습되어 있는 PaliGemma와 같은 Vision-Language Model 위에, 추가 학습을 통해 로봇의 행동을 생성할 수 있는 Action Expert를 결합하는 방식이 일반적입니다. 현재의 Physical AI 시대를 열었다고 평가받는 미국의 Physical Intelligence(PI)사가 공개한 첫 번째 모델 π0[5] 역시 이러한 접근의 대표적인 사례로 볼 수 있습니다. 이 회사는 학습 기반 로보틱스 연구를 선도하는 두 명의 교수, Sergey Levine과 Chelsea Finn이 공동으로 설립한 곳입니다. 2025년 10월 현재, 이 기업의 가치는 약 24억 달러, 한화로 약 2조 7,000억 원 수준으로 알려져 있습니다. 두 교수는 회사를 창업하기 전 구글과 함께 여러 로보틱스 연구를 진행해 왔습니다. 그중 하나가 바로 RT-2[6]로, 제가 아는 한 이 논문에서 처음으로 Vision-Language-Action Model이라는 표현이 사용되었습니다. 당시 구글은 자체 개발한 PaLM-E라는 Vision-Language Model에 로봇 행동 데이터를 토크나이즈(tokenize)하여 모델을 학습시키는 방식을 취했습니다.

Vision-Language-Action(VLA) Model이라는 단어는 상당히 직관적입니다. 로봇이 작업하는 환경을 카메라로 촬영해 시각적 정보를 획득하고, 언어를 통해 작업 지시를 받아 그에 맞는 행동을 생성하는 모델을 의미합니다. 사실 이러한 시도가 완전히 새로운 것은 아닙니다. 로보틱스 분야에서는 오래전부터 Visual Servoing이라고 불리는 개념이 존재했습니다. 이는 실시간 시각 정보를 바탕으로 로봇의 동작을 제어하는 방법론을 가리킵니다. 두 접근의 차이를 살펴보면, 기존의 Visual Servoing은 주로 명시적 규칙에 기반한 rule-based 방식으로 제어를 수행한 반면, VLA Model은 이미 학습된 Vision-Language Model을 활용해 데이터 기반 학습으로 로봇의 행동을 결정한다는 점이 다릅니다. 이러한 변화가 가능해진 이유는, 과거에 비해 시각 정보와 언어 정보를 훨씬 효과적으로 학습할 수 있게 되었고, 동시에 다양한 Foundation Model을 로봇 제어에 응용할 수 있는 환경이 마련되었기 때문입니다.

