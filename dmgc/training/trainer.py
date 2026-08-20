import torch
import torch.nn.functional as F


def kl_divergence(p, q):
    return torch.sum(p * torch.log(p/q))

class GraphTrainer:
    def __init__(self, model, n_networks, n_clusters, u_is, u_js, u_labels, b, cross_networks, cross_networks_masks, config):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.config = config
        self.n_networks = n_networks
        '''
        self.u_is = u_is
        self.u_js = u_js
        self.u_labels = u_labels
        self.b = b
        self.cross_networks = cross_networks
        self.cross_networks_masks = cross_networks_masks
        '''
        self.n_clusters = n_clusters

        self.u_is = [x.to(self.device) if x is not None else None for x in u_is]
        self.u_js = [x.to(self.device) if x is not None else None for x in u_js]
        self.u_labels = [x.to(self.device) if x is not None else None for x in u_labels]
        self.b = [x.to(self.device) if x is not None else None for x in b]
        self.cross_networks = [
            [x.to(self.device) if x is not None else None for x in row]
            for row in cross_networks
        ]

        self.cross_networks_masks = [
            [x.to(self.device) if x is not None else None for x in row]
            for row in cross_networks_masks
        ]



    def getFirstOrderLoss(self,qs):
        first_order =[]

        for i in range(self.n_networks):
            u_i_q = torch.index_select(qs[i],dim=0,index=self.u_is[i])
            u_j_q = torch.index_select(qs[i],dim=0,index=self.u_js[i])
            inner_product = torch.sum(u_i_q*u_j_q,dim=1)
            first_order.append(-torch.mean(F.logsigmoid(self.u_labels[i]*inner_product)))

        '''
        u_i_q = torch.index_select(qs[1], dim=0, index=self.u_is[1])
        u_j_q = torch.index_select(qs[1], dim=0, index=self.u_js[1])
        inner_product = torch.sum(u_i_q * u_j_q, dim=1)
        first_order.append(-torch.mean(F.logsigmoid(self.u_labels[1]*inner_product)))

        '''
        #print('1',first_order[0])
        #print('2',first_order[1])
        #weight = torch.tensor([1.0, 1.0, 1.0, 2.5],device=self.device)
        firstOrderLoss = torch.sum(torch.stack(first_order))
        return firstOrderLoss


    def getVariationalLowerBoundLoss(self,vl):

        variationalLowerBoundLoss = torch.sum(torch.stack(vl))
        return variationalLowerBoundLoss


    def getSecondOrderLoss(self,a,x):
        x = [x.to(self.device) for x in x]
        second_order = []

        for i in range(self.n_networks):
            diff = (a[i]-x[i])*self.b[i]
            second_order.append(torch.mean(torch.pow(diff,2),dim=[0,1]))
        '''
        diff = (a[1]-x[1])*self.b[1]
        second_order.append(torch.mean(torch.pow(diff, 2), dim=[0, 1]))
        '''
        secondOrderLoss= torch.sum(torch.stack(second_order))
        return secondOrderLoss

    def getCrossLoss(self,qs,Cs):

        losses = []
        for i in range(self.n_networks):
            for j in range(self.n_networks):
                if i==j:
                    continue
                tqj = qs[j].T
                cij_tqj = torch.matmul(Cs[i][j],tqj)
                cij_tqj_t = cij_tqj.T
                sij_cij_tqj_t = torch.matmul(self.cross_networks[i][j], cij_tqj_t)
                '''
                row_min = sij_cij_tqj_t.min(dim=1, keepdim=True).values
                row_max = sij_cij_tqj_t.max(dim=1, keepdim=True).values
                denominator = (row_max - row_min).clamp(min=1e-8)
                sij_cij_tqj_t = (sij_cij_tqj_t - row_min) / denominator
                '''
                diff = qs[i] - sij_cij_tqj_t


                #diff = F.kl_div(qs[i].log(),sij_cij_tqj_t, reduction='batchmean')

                #print(qs[i])
                #print(sij_cij_tqj_t)
                mask_diff = torch.multiply(self.cross_networks_masks[i][j], diff)

                losses.append(torch.sum(torch.pow(mask_diff, 2)))
                #losses.append(torch.mean(torch.abs(mask_diff)))
        #print(losses)


        cross_loss = torch.sum(torch.stack(losses))

        return cross_loss

    def getClusterLoss(self,qs):
        entropy_losses = []
        uniform_losses = []


        for i in range(self.n_networks):
            entropy = torch.sum(qs[i]*qs[i],dim=1)
            elosses = -torch.mean(F.logsigmoid(entropy))
            entropy_losses.append(elosses)

            qave = torch.mean(qs[i],dim=0)


            uniform = torch.full([self.n_clusters[i]], 1.0 / self.n_clusters[i]).to(self.device)
            #uniform = torch.tensor([1.25/9.0, 2.25/9.0, 2.25/9.0, 3.25/9.0], device=self.device)

            uniform_losses.append(kl_divergence(qave,uniform))


        #weights = [1.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        #entropy_losses = [w * l for w, l in zip(weights, entropy_losses)]

        entropy_loss = torch.sum(torch.stack(entropy_losses))
        uniform_loss = torch.sum(torch.stack(uniform_losses))


        cluster_loss = entropy_loss + self.config.l01 * uniform_loss
        return cluster_loss,entropy_loss,uniform_loss

    def getLoss(self, qs, a, x, Cs, epoch): #fnn
    #def getLoss(self, qs, Cs,vl): #gnn

        firstOrderLoss = self.getFirstOrderLoss(qs)
        secondOrderLoss = self.getSecondOrderLoss(a,x)  #fnn
        #variationalLowerBoundLoss = self.getVariationalLowerBoundLoss(vl) #gnn

        crossLoss = self.getCrossLoss(qs,Cs)

        clusterLoss,entropyLoss, uniformLoss = self.getClusterLoss(qs)

        orderLoss = firstOrderLoss + self.config.l11 * secondOrderLoss #fnn

        #orderLoss = firstOrderLoss
        #orderLoss = firstOrderLoss +args.l11*variationalLowerBoundLoss #gnn

        Loss = self.config.l0 * clusterLoss + self.config.l1 * orderLoss + self.config.l2 * crossLoss



        return Loss,firstOrderLoss,secondOrderLoss,crossLoss,clusterLoss,entropyLoss,uniformLoss #fnn



    #return Loss, firstOrderLoss, variationalLowerBoundLoss, crossLoss, clusterLoss, entopyLoss, uniformLoss #gnn

































