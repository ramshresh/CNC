base_in_dir = r"C:\work\CNC\3D Maps\Data\NSET_data_prep\OBJ_files"
in_fname = r"ly1_3D Roughing 1_top_right_origin.backup.tap"
out_fname =in_fname+"-xyz_values.txt"
import re, os

in_fpath = os.path.join(base_in_dir, in_fname)
out_fpath = os.path.join(base_in_dir, out_fname)

#  Meta Characters -->  [] . ^ $ * + ? {} () \ |
#  search_regex = r".*z.*\n"  for python: match: "^.*Z.*$\n" , replace: "Z-", "Z"

"""
G+
[-+]? # optional sign
(?:
    (?:  \d* \. \d+ )         # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?:  \d+ \.? )            # 1. 12. 123. etc 1 12 123 etc
)
# followed by optional exponent part if desired
(?: [Ee] [+-]? \d+ ) ?
"""


numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'

numeric_const_pattern_G = r"""G+[-+]? (?: (?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""
numeric_const_pattern_M = r"""M+ [-+]? (?:(?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""
numeric_const_pattern_T = r"""T+ [-+]? (?:(?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""
numeric_const_pattern_S = r"""S+ [-+]? (?:(?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""
numeric_const_pattern_X = r"""X+ [-+]? (?:(?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""
numeric_const_pattern_Y = r"""Y+ [-+]? (?:(?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""
numeric_const_pattern_Z = r"""Z+ [-+]? (?:(?:  \d* \. \d+ )|(?:  \d+ \.? ))(?: [Ee] [+-]? \d+ ) ?"""

rx= re.compile(numeric_const_pattern, re.VERBOSE)
rx_G= re.compile(numeric_const_pattern_G, re.VERBOSE)
rx_M= re.compile(numeric_const_pattern_M, re.VERBOSE)
rx_T= re.compile(numeric_const_pattern_T, re.VERBOSE)
rx_S= re.compile(numeric_const_pattern_S, re.VERBOSE)
rx_X= re.compile(numeric_const_pattern_X, re.VERBOSE)
rx_Y= re.compile(numeric_const_pattern_Y, re.VERBOSE)
rx_Z= re.compile(numeric_const_pattern_Z, re.VERBOSE)
out_data = ""
count=20
with open(in_fpath, 'r') as in_f:
    lines = in_f.readlines()
    for l in lines:
        """
        matched_z = re.match("^.*Z.*$\n", l)
        if(matched_z):
            #print ("{}  --> {}".format(l,re.findall(r"[-+]?\d*\.\d+|\d+", l)))
            print ("{}  --> {}".format(l,rx.findall(l)))
        """


        count = count-1
        if(count>0):
            g_str_list = rx_G.findall(l)
            m_str_list = rx_M.findall(l)
            t_str_list = rx_T.findall(l)
            s_str_list = rx_S.findall(l)
            x_str_list = rx_X.findall(l)
            y_str_list = rx_Y.findall(l)
            z_str_list = rx_Z.findall(l)

            g_str_0= g_str_list[0] if (len(g_str_list)>0) else None
            m_str_0= m_str_list[0] if (len(m_str_list)>0) else None
            t_str_0= t_str_list[0] if (len(t_str_list)>0) else None
            s_str_0= s_str_list[0] if (len(s_str_list)>0) else None
            x_str_0= x_str_list[0] if (len(x_str_list)>0) else None
            y_str_0= y_str_list[0] if (len(y_str_list)>0) else None
            z_str_0= z_str_list[0] if (len(z_str_list)>0) else None
            

            g_val= rx.findall(g_str_0)[0] if (g_str_0 is not None and len(rx.findall(g_str_0))>0) else None
            m_val= rx.findall(m_str_0)[0] if (m_str_0 is not None and len(rx.findall(m_str_0))>0) else None
            t_val= float(rx.findall(t_str_0)[0]) if (t_str_0 is not None and len(rx.findall(t_str_0))>0) else None
            s_val= float(rx.findall(s_str_0)[0]) if (s_str_0 is not None and len(rx.findall(s_str_0))>0) else None
            x_val= float(rx.findall(x_str_0)[0]) if (x_str_0 is not None and len(rx.findall(x_str_0))>0) else None
            y_val= float(rx.findall(y_str_0)[0]) if (y_str_0 is not None and len(rx.findall(y_str_0))>0) else None
            z_val= float(rx.findall(z_str_0)[0]) if (z_str_0 is not None and len(rx.findall(z_str_0))>0) else None
            
  
            print ("{}  --> G: {} M: {} T: {} S: {} X : {}  Y : {} Z: {} ".format(l,g_val,m_val,t_val,s_val,x_val,y_val,z_val))
        

with open(out_fpath, "w") as out_f:
    out_f.write(out_data)
