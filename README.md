# fluxx
AI framework, c++ &amp;  python 1: 1 matching, powerfull time series neural net, chatbot including

Time series models that include: Spiking model using som(self-organizing map), quauntum attention model, Bayesian inference network using rdbms
Requirement
----------------------------
windows os  
cuda 11.8  
libtorch 2.1.1  
>https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.1%2Bcu118.zip  
>링크 다운로드 후 설치할 위치에 압축 해제

anaconda 24.5.0  
python 3.9.13  
numpy 1.26.4  
mpi4py  
 
morphic.py 프로그램의 os.add_dll_directory("E:\\down\libtorch\lib")에서 라이브러리 위치를 위 압축 해제 폴더 위치로 변경한다.  

-----------------------------

sign curve 
  training
  
    python hsign_regress.py 1 0 0 sign_regress 1 1     #초기화
    python hsign_regress.py 5 0 0 sign_regress 0 50    #50번 에포크 수행
  test
  
    python hsign_regress.py 6 0 0 sign_regress 0


chatbot

  korqbot #
  
    korq_set.zip 압축해제후
    
    training
      python korqbot.py --case 1 --m_name korqbot --d_name korq_set  #초기화 & 학습
      python korqbot.py --case 5 --m_name korqbot --d_name korq_set  #이어서 학습
      
    test
      python korqbot.py --case 6 --m_name korqbot --d_name korq_set  #추론 테스트


anormal dection ( 이상탐지 )

링크에서 data 다운받아 anormaldata 폴더에 압축해제후 실행 https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras

  training
  
    python hanormal.py 1 0 0 anormal 1 1 #초기화
    python hanormal.py 5 0 0 anormal 0 50 #50번 에포크 수행

  test
  
    python hanormal.py 6 0 0 anormal 0 
