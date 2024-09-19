# def unzip_command(zip_file,dst_path):
#     """처음 읽은 파일이 zip파일이기 때문에 압축해제를 진행하는 함수

#     Args:
#         zip_file (str): 압축할 파일
#         dst_path (str): 압축한 결과를 저장할 폴더

#     Returns:
#         str: 압축한 결과를 저장할 폴더 디렉토리 명 반환
#     """    
#     if not os.path.exists(dst_path):
#         os.makedirs(dst_path)

#     os.system("unzip "+zip_file+" -d "+dst_path)

#     return dst_path

# def unzip_file(file_list):
#     """unzip_command를 활용해서 실제로 데이터를 압축해제하는 함수

#     Args:
#         file_list (list): zip파일이 적혀있는 리스트 변수 (train,valid)

#     Returns:
#         list: 압축해제한 폴더들의 경로를 저장하는 변수(train,valid)
#     """    
#     unzip_folders=[]
#     for file in tqdm(file_list,desc="unzipping"):
#         if file.endswith('.zip'):
#             unzipped_folder=os.path.splitext(file)[0]
#             folder=unzip_command(file,unzipped_folder)
#             unzip_folders.append(folder)

    
#     return unzip_folders
