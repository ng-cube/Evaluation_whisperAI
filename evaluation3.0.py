import io
import time
import whisper
import soundfile as sf
import numpy as np
import pandas as pd

from pydub import AudioSegment
from soundfile import SEEK_END,SoundFile
from util import split_files_wo_save,save_split_files
from evaluation import tgrid2col,calc_wer_score,df2csv,normalise_text,removechar

# changes made: reorganize the code (to functions),normalise the texts

def GT2df(gt_tgrid_file):
    try:
        gt_df = tgrid2col(gt_tgrid_file,fileEncoding="utf-16") 
    except UnicodeError:
        gt_df = tgrid2col(gt_tgrid_file,fileEncoding="utf-8")    
    return gt_df


def pred_segments(gt_df,audio_file,model,file_path):
    '''split to clips according to the ground_truth and use whisper to gen text '''
    #save_split_files(gt_df,audio_file,file_path)  
    splited_audio = split_files_wo_save(gt_df,audio_file) # tested,dutation is the same as gt

    counter = 0
    res_segments = []
    for audio in splited_audio:
        with io.BytesIO() as buffer:
            audio_data = audio.export(out_f=buffer,format="wav")
            buffer.seek(0)
            buffer_data = buffer.read() 
        buffer_data = io.BytesIO(buffer_data)
        audio_data,sample_rate = sf.read(buffer_data,dtype="float32")

        #audio_data = np.array(audio_data)
        try:
            results = model.transcribe(audio_data,language="en")
            res_segments.append(results['segments'])           
        except RuntimeError as e:
            print("RuntimeError!",e)   
        
        counter += 1
        if counter == 6:
            break
    '''res_segments are the results returned by prediction'''
    return  res_segments


def pred_segments_new(gt_df,audio_file,model):
    audio = AudioSegment.from_file(audio_file,format="wav")
    segment_lengths = gt_df["duration"].values.tolist()
    res_segments = []

    for i,segment_length in enumerate(segment_lengths):
        segments = audio[:segment_length]

        for j in range(segments.channels):
            with io.BytesIO() as buffer:
                samples = np.array(segments.get_array_of_samples()[j::segments.channels])      #.astype(np.float64) / 2**(segment.sample_width * 8 - 1) 
                max_sample = 2**(segments.sample_width * 8 - 1) - 1
                audio_data = np.array(samples,dtype=np.float32) / max_sample
                #audio_data = audio_data.reshape(-1,segments.channels)
                audio_data = audio_data.astype(np.float32)
                sf.write(buffer,audio_data.T,segments.frame_rate,format='wav',subtype='float') #.astype(np.float32)
                buffer.seek(0)
                buffer_data = buffer.read()
        
        buffer_data = io.BytesIO(buffer_data)
        audio_data,sample_rate = sf.read(buffer_data,dtype="float32")

        try:
            results = model.transcribe(audio_data,language="en")
            res_segments.append(results['segments'])           
        except RuntimeError as e:
            print("RuntimeError!",e)   
    
    return res_segments


def write_prediction(res_segments,gt_df):
    res_col = ['id','start','end','Predicted_text']
    res = []
    for count,result in enumerate(res_segments):
        try:
            res.append([count,
                        result[0]['start']*1000,
                        result[0]['end']*1000,
                        result[0]['text']])
        except IndexError:
            print("empty list!")
            res.append([None,None,None,None])
            pass       
    res_df = pd.DataFrame(res,columns=res_col)

    overalldf = pd.merge(gt_df, res_df, how="outer", on=["id"])
    overalldf = overalldf.fillna("-")
    
    return overalldf


def evaluation(overalldf):
    df = overalldf
    df['GT_text'] = df['GT_text'].astype('string')
    df["Predicted_text"] = df['Predicted_text'].astype('string')
    df["Predicted_text"] = df['Predicted_text'].apply(normalise_text)
    df["Predicted_text"] = df['Predicted_text'].apply(removechar)
    
    score_list = []   
    for row in df.itertuples(): 
        try:
            score = calc_wer_score(row.GT_text,row.Predicted_text)
            score_list.append(score)
        except ValueError:
            print(ValueError)
            score_list.append(None)
    
    df.insert(8,"wer",score_list)
    return df


def main():
    root_path = "/mnt/6T-storage/IMDA/PART3"
    audio_file = root_path + "/Audio_Same_CloseMic/3000-1.wav"   
    gt_tgrid_file = root_path + "/Scripts_Same/3000-1.TextGrid"
    
    model_type = "base"
    model = whisper.load_model(model_type)

    start_time = time.time()

    gt_df = GT2df(gt_tgrid_file)   
    res_segments = pred_segments(gt_df,audio_file,model,root_path) 
    print(res_segments)
    print("Done with prediction!")
    
    # overalldf = write_prediction(res_segments,gt_df)
    # overalldf = evaluation(overalldf)
    # print(overalldf)

    # df2csv(overalldf,root_path,"/{}_res.csv".format("3000-1"))
    # print("success!")
    
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()


