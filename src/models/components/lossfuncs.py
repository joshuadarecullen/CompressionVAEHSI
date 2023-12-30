import torch
import torch.nn.functional as F

def sam_loss(original, reconstruction):
    # Flatten the spectral dimension
    original_flat = original.view(original.size(0), -1)
    reconstruction_flat = reconstruction.view(reconstruction.size(0), -1)

    # Normalize the vectors (to unit vectors)
    original_norm = F.normalize(original_flat, p=2, dim=1)
    reconstruction_norm = F.normalize(reconstruction_flat, p=2, dim=1)

    # Calculate the SAM for each pixel
    sam_angles = torch.acos((original_norm * reconstruction_norm).sum(dim=1))

    # Average SAM over all pixels
    return sam_angles.mean()


def sid_loss(original, reconstruction):
    # Ensure no zero elements for log stability
    eps = 1e-10
    original += eps
    reconstruction += eps

    # Normalize the spectral vectors
    original_norm = original / original.sum(dim=1, keepdim=True)
    reconstruction_norm = reconstruction / reconstruction.sum(dim=1, keepdim=True)

    # Calculate SID for each pixel
    kl_orig_recon = (original_norm * (original_norm / reconstruction_norm).log()).sum(dim=1)
    kl_recon_orig = (reconstruction_norm * (reconstruction_norm / original_norm).log()).sum(dim=1)

    sid = kl_orig_recon + kl_recon_orig

    # Average SID over all pixels
    return sid.mean()


if __name__ == "__main__":
    # Example Spectral Angle Mapper (SAM) usage in a VAE training loop
    '''
    vae_loss = reconstruction_loss + kl_divergence  # Your existing VAE loss components
    sam = sam_loss(original_images, reconstructed_images)
    total_loss = vae_loss + alpha * sam  # where alpha is a weighting factor
    '''

    # Example Spectral Information Divergence (SID) usage in a VAE training loop
    '''
    vae_loss = reconstruction_loss + kl_divergence  # Your existing VAE loss components
    sid = sid_loss(original_images, reconstructed_images)
    total_loss = vae_loss + alpha * sid  # where alpha is a weighting factor
    '''

