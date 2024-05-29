
윈도우에서 아나콘다 conda 22.9.0 설치후 수행

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
