# tensoRflow ML-lab-08
# Sung Kim 교수님 강의 소스: https://youtu.be/ZYX0FaqUeN4?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
library(magrittr)

sess <- tf$InteractiveSession()

# 오브젝트 선언 실험
matrix1 <- tf$constant(matrix(c(1:6), ncol = 2))
matrix2 <- tf$constant(matrix(c(1:6), ncol = 2), dtype = "float32")
matrix3 <- tf$constant(matrix(as.numeric(c(1:6)), ncol = 2))

# 결과값 확인
matrix1   # int32
matrix2   # float32
matrix3   # float64

# 결과값 해석
# Tensor("Const_1:0", shape=(3, 2), dtype=float32) 의미
# 두번째로 정의된 constant 오브젝트이며, 반환값이 1
# (3행, 2열) 모양이며, 원소타입: 32비트 실수
# 그래프 리셋방법: tf$reset_default_graph()

# 실제값 확인 => R에서는 다 똑같이 보임
matrix1$eval()
matrix2$eval()
matrix3$eval()

# 함수 적용방향 실험
# 실험 대상 3D array: 3 행 4 열
matrix1 <- aperm(array(c(1:24), dim = c(4, 3, 2)), perm = c(2, 1, 3)) %>% 
           tf$constant()
matrix1$eval()
# tensoRfolw에서는 0L 세로방향, 1L 가로방향, 2L 레이어방향
# 참고: parameter 뒤에는 항상 정수를 의미하는 L을 붙여줌.

# 3D array 기준: 
tf$reduce_sum(matrix1, axis = 0L)$eval()  # 세로방향
tf$reduce_sum(matrix1, axis = 1L)$eval()  # 가로방향
tf$reduce_sum(matrix1, axis = 2L)$eval()  # 레이어방향

# -의 의미는 정의된 방향 중 '뒤에서부터'를 의미함.
# 즉, -1L은 뒤에서부터 첫번째 방향인 레이어방향
tf$reduce_sum(matrix1, axis = -1L)$eval() # 레이어방향
tf$reduce_sum(matrix1, axis = -2L)$eval() # 가로방향
tf$reduce_sum(matrix1, axis = -3L)$eval() # 세로방향

# Note: 방향 없으면 벡터취급
tf$reduce_sum(matrix1)$eval()

tf$shape(matrix1) # 행,열,레이어 순서

# tensor reshape: -1L의 의미는 "알아서" 하라는 의미.
# 다음 명령어의 의미는 2D 행렬로 변환하되,
# 열수는 4개, 행갯수는 알아서 바꿔라라는 의미.
matrix2 <- tf$reshape(matrix1, shape = c(-1L, 4L))
matrix2$eval() # 우리의 예상대로 나오지 않음.

# 친근한 R함수 사용하기
matrix3 <- rbind(matrix1[,,0]$eval(), matrix1[,,1]$eval()) %>%
           tf$constant(dtype = "float32")
matrix3$eval()

# tensoRflow 함수만 이용하기
matrix4 <- tf$concat(c(matrix1[,,0], matrix1[,,1]), axis = 0L)
matrix4$eval()

# squeeze와 expand_dim 함수
matrix4 <- tf$expand_dims(tf$transpose(matrix4), 0L)
matrix4$eval()
matrix4 <- tf$squeeze(tf$transpose(matrix4))
matrix4$eval()

matrix4 <- tf$expand_dims(matrix4, 1L)
matrix4$eval()
matrix4 <- tf$squeeze(matrix4)
matrix4$eval()

# onehot coding
onehot <- tf$one_hot(as.integer(c(0,1,2,0)), depth = 3L)
onehot$eval()

# Close session
sess$close()
