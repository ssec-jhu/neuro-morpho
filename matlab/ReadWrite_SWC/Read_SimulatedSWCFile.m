%This function reads a swc file and returns a tree structure
%%% Assumes a binary tree
% The tree structure contains the following variables:
% Tree = struct with fields:
%            AllBranchPointIDs: N×1  branchpoint IDs in the input ID file; N being the number of branchpoints
%            AllEndPointIDs: N×1 endhpoint IDs in the input ID file; N being the number of endpoints
%            AllBranchPointCoordinate: Nx3 branchpoint coordinates in the swc file
%            AllEndPointCoordinate: N×3 branchpoint coordinates in the swc file
%            Subtree: NX1 structure contains the subtrees coming out from the root
%                   Subtree(1).Branch: struct that contains the following information about this subtree
%                                       N: Number of points along the branch
%                                       length: the length in the branch (whatever unit is in swc file
%                                       x: Nx1 x coordinates along the branch
%                                       y: Nx1 y coordinates along the branch
%                                       z: Nx1 z coordinates along the branch
%                                       R: Nx1 radius along the branch
%                                       type: Type of branch (1-> terminal ends in a tip; 2->internal end in a branchpoint)
%%usage 
%    Tree=Read_SimulatedSWCFile('Input_swc_file.swc');
%% Read an SWC file and reconstruct subtree structures
function Tree=Read_SimulatedSWCFile(filename)
    global Branch NBranch;    
    % Read file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Error opening file');
    end
    
    n = 0;
    N_BranchInitial = 0;
    nbrpts=0;
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, '#')
            continue;
        end
        
        data = sscanf(line, '%d %d %f %f %f %f %d');
        if numel(data) == 7
            n = n + 1;
            ID(n,1) = data(1);
            BranchID(n,1) = data(2);
            X(n,1) = data(3);
            Y(n,1) = data(4);
            Z(n,1)=data(5);
            Radius(n,1) = data(6);
            ParentID(n,1) = data(7);
            
            if ParentID(n) == 1
                N_BranchInitial = N_BranchInitial + 1;
                InitialInd(N_BranchInitial) = ID(n);
            end
            %%%%%%%%%%%%%%%%%% Identify Branchpoints
            if ID(n) - ParentID(n) ~= 1 && ParentID(n) ~= 1 && ParentID(n) ~= -1
                Label(ParentID(n)) = 2;
                nbrpts=nbrpts+1;
                Tree.AllBranchPointIDs(nbrpts,1)=ParentID(n);
                Tree.AllBranchPointCoordinate(nbrpts,1:3)=[X(ParentID(n)),Y(ParentID(n)),Z(ParentID(n))];
            end
        end
    end
    fclose(fid);    


    %%%%%%%%%%%%%%%%%%%%%% Identify end points
    nend = 0;
    for i = 1:n
        if ~ismember(ID(i), ParentID)
            nend = nend + 1;
            Label(ID(i)) = 1;
            Tree.AllEndPointIDs(nend,1)=ID(i);
            Tree.AllEndPointCoordinate(nend,1:3)=[X(ID(i)),Y(ID(i)),Z(ID(i))];
        end 
        ChildPtID(i,1:2) = [0,0];
    end
    
    %%%%%%%%%%%%%%% Determine child IDs at branch points
    for i = 1:n
        if Label(i) == 2
            nchild = 0;
            for j = 1:n
                if ParentID(j) == i
                    nchild = nchild + 1;
                    ChildPtID(i, nchild) = j;
                    
                end
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Reconstruct subtrees
    XYZR=[X,Y,Z,Radius];
 
    for i = 1:N_BranchInitial
        Branch=struct([]);
        NBranch = 0;
        make_subtree(InitialInd(i),XYZR,Label,ChildPtID);
        subtree(i).Branch=Branch;
    end
    Tree.Subtree=subtree;
end
%% recursive function to create the subtree from the swc file
    function make_subtree(np,XYZR,Label,ChildPtID)
    % Recursive function to construct subtrees
     global Branch NBranch;

    NBranch = NBranch + 1;
    Branch(NBranch).N = 0; 
    Branch(NBranch).length=0;
    while Label(np) == 0
        Branch(NBranch).N= Branch(NBranch).N + 1;
        Branch(NBranch).x(Branch(NBranch).N,1)=XYZR(np,1);
        Branch(NBranch).y(Branch(NBranch).N,1)=XYZR(np,2);
        Branch(NBranch).z(Branch(NBranch).N,1)=XYZR(np,3);
        Branch(NBranch).R(Branch(NBranch).N,1)=XYZR(np,4);
       
         if Branch(NBranch).N > 1
            Branch(NBranch).length = Branch(NBranch).length + sqrt((Branch(NBranch).x(end)-Branch(NBranch).x(end-1))^2 +...
                (Branch(NBranch).y(end)-Branch(NBranch).y(end-1))^2 + (Branch(NBranch).z(end)-Branch(NBranch).z(end-1))^2);
         end
        np = np + 1;
    end
    
    if Label(np) == 2
        Branch(NBranch).type=2;
        Branch(NBranch).N= Branch(NBranch).N + 1;
        Branch(NBranch).x(Branch(NBranch).N,1)=XYZR(np,1);
        Branch(NBranch).y(Branch(NBranch).N,1)=XYZR(np,2);
        Branch(NBranch).z(Branch(NBranch).N,1)=XYZR(np,3);
        Branch(NBranch).R(Branch(NBranch).N,1)=XYZR(np,4);
        Branch(NBranch).length = Branch(NBranch).length + sqrt((Branch(NBranch).x(end)-Branch(NBranch).x(end-1))^2 +...
                (Branch(NBranch).y(end)-Branch(NBranch).y(end-1))^2 + (Branch(NBranch).z(end)-Branch(NBranch).z(end-1))^2);
        
        make_subtree(ChildPtID(np, 1),XYZR,Label,ChildPtID);
        make_subtree(ChildPtID(np, 2),XYZR,Label,ChildPtID);
    else
        Branch(NBranch).type=1;
    end
end
